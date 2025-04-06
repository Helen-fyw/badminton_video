import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import json
from ultralytics import YOLO  # Replace YOLOv5 import with YOLOv8
import logging

class BadmintonDataset(Dataset):
    def __init__(self, root_dir: str, video_dir: str, transform=None, sequence_length: int = 16):
        """
        初始化数据集
        Args:
            root_dir: ShuttleSet数据集根目录
            video_dir: 视频文件目录
            transform: 数据增强转换
            sequence_length: 每个样本包含的帧数
        """
        self.root_dir = root_dir
        self.video_dir = video_dir
        self.transform = transform  # 数据增强转换
        self.sequence_length = sequence_length  # 每个样本包含的帧数
        
        # 读取所有比赛信息
        self.matches = pd.read_csv(os.path.join(root_dir, 'match.csv'))
        self.homography = pd.read_csv(os.path.join(root_dir, 'homography.csv'))
        
        # 击球类型映射
        self.shot_type_map = {
            '發短球': 1, '長球': 2, '推球': 3, '殺球': 4, '擋小球': 5,
            '平球': 6, '放小球': 7, '挑球': 8, '切球': 9, '發長球': 10,
            '接不到': 11, '勾球': 12, '點扣': 13, '防守回抽': 14,
            '過度切球': 15, '撲球': 16, '後場抽平球': 17, '未知球種': 18
        }

        print("\n数据集中的比赛：")
        for _, match in self.matches.iterrows():
            video_path = os.path.join(self.video_dir, match['video'], match['video'] + '.mp4')
            if os.path.exists(video_path):
                print(f"✓ {match['video']}")
            else:
                print(f"✗ {match['video']}")
        
        # 收集所有样本
        self.samples = self._collect_samples()
        self.yolo_model = YOLO("yolov8n-pose.pt")  # Load YOLOv8 model for pose estimation
        self.yolo_model.overrides['verbose'] = False  # 禁用YOLO的详细输出
        
        # 过滤无效样本
        invalid_sample_count = 0
        valid_samples = []
        for sample in tqdm(self.samples, desc="过滤无效样本"):
            _, valid = self._load_frames(sample['video_path'], sample['frame_num'], sample['player'])
            if valid:
                valid_samples.append(sample)
            else:
                logging.warning(f"样本无效，已删除: {sample}")
                invalid_sample_count += 1
        self.samples = valid_samples

        # 打印无效样本数量
        print(f"无效样本数量: {invalid_sample_count}")
        logging.info(f"无效样本数量: {invalid_sample_count}")
        
    def _collect_samples(self) -> List[Dict]:
        """收集所有训练样本"""
        samples = []
        for _, match in tqdm(self.matches.iterrows(), total=len(self.matches)):
            match_id = match['video']
            video_path = os.path.join(self.video_dir, match_id, match_id + '.mp4')
            
            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                continue
            
            match_dir = os.path.join(self.root_dir, match_id)
            if not os.path.exists(match_dir):
                print(f"比赛目录不存在: {match_dir}")
                continue
                
            for set_num in range(1, 4):
                set_file = os.path.join(match_dir, f'set{set_num}.csv')
                if not os.path.exists(set_file):
                    continue
                    
                set_data = pd.read_csv(set_file)
                for i in range(len(set_data)):
                    shot_type = set_data.iloc[i]['type']
                    if shot_type not in self.shot_type_map:
                        continue
                        
                    frame_num = set_data.iloc[i]['frame_num']
                    player = set_data.iloc[i]['player']
                    sample = {
                        'video_path': video_path,
                        'frame_num': frame_num,
                        'shot_type': self.shot_type_map[shot_type],
                        'sequence_length': self.sequence_length,
                        'player': player
                    }
                    samples.append(sample)
                    
        if not samples:
            raise ValueError("没有找到有效的样本！请检查数据集路径和视频文件是否正确。")
                    
        return samples
    
    def _pad_frames(self, frames: List[np.ndarray], sequence_length: int, resize_size: Tuple[int, int]) -> List[np.ndarray]:
        """Pad frames to match the required sequence length."""
        if len(frames) < sequence_length:
            frames.extend([frames[-1]] * (sequence_length - len(frames)))
        return frames

    def _load_frames(self, video_path: str, frame_num: int, player: str, resize_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, bool]:
        """加载视频帧并裁剪 YOLO 检测框内的人体区域"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"无法打开视频文件: {video_path}")
            
            frames = []
            start_frame = max(0, frame_num - self.sequence_length)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for _ in range(self.sequence_length):
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"视频帧读取失败: {video_path}, 帧号: {start_frame}")
                    break
                frames.append(frame)
            
            cap.release()
            frames = self._pad_frames(frames, self.sequence_length, resize_size)

            # Batch YOLO inference
            results = self.yolo_model(frames, stream=False)
            processed_frames = []
            valid_frame_detected = False

            for idx, result in enumerate(results):
                selected_bbox = None
                for box_idx, bbox in enumerate(result.boxes.xyxy.cpu().numpy()):
                    if (player == 'B' and box_idx == 0) or (player == 'A' and len(result.boxes) > 1 and box_idx == 1):
                        selected_bbox = bbox
                        break
                
                if selected_bbox is None:
                    processed_frames.append(np.zeros((3, resize_size[0], resize_size[1]), dtype=np.uint8))
                else:
                    x1, y1, x2, y2 = map(int, selected_bbox)
                    cropped_frame = frames[idx][y1:y2, x1:x2]
                    cropped_frame = cv2.resize(cropped_frame, resize_size)
                    processed_frames.append(np.transpose(cropped_frame, (2, 0, 1)))
                    valid_frame_detected = True

            return np.array(processed_frames), valid_frame_detected
        except Exception as e:
            logging.error(f"加载视频帧时出错: {e}, 视频路径: {video_path}, 帧号: {frame_num}")
            return None, False
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            sample = self.samples[idx]
            if 'video_path' not in sample or 'frame_num' not in sample or 'player' not in sample:
                raise KeyError(f"Sample is missing required keys: {sample}")
            
            frames, _ = self._load_frames(sample['video_path'], sample['frame_num'], sample['player'])
            if frames is None or len(frames) == 0:
                logging.warning(f"加载帧失败，返回空样本: {sample}")
                return torch.zeros((3, self.sequence_length, 224, 224)), torch.tensor(0)
                
            frames = torch.from_numpy(frames).float()
            frames = frames.permute(1, 0, 2, 3)
            
            if self.transform:
                frames = self.transform(frames)
                
            return frames, torch.tensor(sample['shot_type'])
        except Exception as e:
            logging.error(f"获取样本时出错: {e}, 索引: {idx}")
            return torch.zeros((3, self.sequence_length, 224, 224)), torch.tensor(0)

def create_data_loaders(root_dir: str, video_dir: str, batch_size: int = 2, num_workers: int = 1, transform=None):
    """
    创建数据加载器
    Args:
        root_dir: 数据集根目录
        video_dir: 视频文件目录
        batch_size: 每个批次的样本数量
        num_workers: 数据加载的子进程数量
        transform: 数据增强转换
    """
    dataset = BadmintonDataset(root_dir, video_dir, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Enable pin_memory for faster data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

if __name__ == '__main__':
    root_dir = 'ShuttleSet'
    video_dir = 'youtube_video'
    train_loader, val_loader = create_data_loaders(root_dir, video_dir)
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    for frames, labels in train_loader:
        print(f"批次形状: {frames.shape}")
        print(f"标签形状: {labels.shape}")
        break