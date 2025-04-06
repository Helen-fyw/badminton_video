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
        self.transform = transform #这是一个数据增强的转换函数，用于对图像进行预处理，如调整大小、归一化等。
        self.sequence_length = sequence_length #这是一个整数，表示每个样本包含的帧数。
        
        # 读取所有比赛信息
        self.matches = pd.read_csv(os.path.join(root_dir, 'match.csv'))
        self.homography = pd.read_csv(os.path.join(root_dir,'homography.csv'))
        
        # TODO 更新击球类型映射，扩展到18类
        self.shot_type_map = {
            '發短球': 1,
            '長球': 2,
            '推球': 3,
            '殺球': 4,
            '擋小球': 5,
            '平球': 6,
            '放小球': 7,
            '挑球': 8,
            '切球': 9,
            '發長球': 10,
            '接不到': 11,
            '勾球': 12,  # 新增类型
            '點扣': 13,  # 新增类型
            '防守回抽': 14,  # 新增类型
            '過度切球': 15,  # 新增类型
            '撲球': 16,  # 新增类型
            '後場抽平球': 17,  # 新增类型
            '未知球種': 18  # 新增类型
        }

        print("\n数据集中的比赛：")
        for _, match in self.matches.iterrows():
            video_path = os.path.join(self.video_dir,match['video'], match['video'] + '.mp4')
            if os.path.exists(video_path):
                print(f"✓ {match['video']}")
            else:
                print(f"✗ {match['video']}")
        
        # 收集所有样本
        self.samples = self._collect_samples()
        self.yolo_model = YOLO("yolov8n-pose.pt")  # Load YOLOv8 model for pose estimation
        
        # 过滤无效样本
        valid_samples = []
        for sample in tqdm(self.samples, desc="过滤无效样本"):
            _, valid = self._load_frames(sample['video_path'], sample['frame_num'], sample['player'])
            if valid:
                valid_samples.append(sample)
            else:
                logging.warning(f"样本无效，已删除: {sample}")
        self.samples = valid_samples
        
    def _collect_samples(self) -> List[Dict]:
        """收集所有训练样本"""
        samples = []
        
        for _, match in tqdm(self.matches.iterrows(), total=len(self.matches)):
            match_id = match['video']
            video_path = os.path.join(self.video_dir, match_id,match_id + '.mp4')
            
            if not os.path.exists(video_path):
                print(f"视频文件不存在: {video_path}")
                continue
            
            # 读取每个set的数据
            match_dir = os.path.join(self.root_dir, match_id)
            if not os.path.exists(match_dir):
                print(f"比赛目录不存在: {match_dir}")
                continue
                
            for set_num in range(1, 4):
                set_file = os.path.join(match_dir, f'set{set_num}.csv')
                if not os.path.exists(set_file):
                    continue
                    
                set_data = pd.read_csv(set_file)
                
                # 处理每个击球
                for i in range(len(set_data)):
                    # 获取击球类型
                    shot_type = set_data.iloc[i]['type']
                    if shot_type not in self.shot_type_map:
                        continue
                        
                    # 获取帧号和击球方
                    frame_num = set_data.iloc[i]['frame_num']
                    player = set_data.iloc[i]['player']
                    
                    # 构建样本
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
    
    def _load_frames(self, video_path: str, frame_num: int, player: str, resize_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, bool]:
        """加载视频帧并裁剪 YOLO 检测框内的人体区域"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"无法打开视频文件: {video_path}")
            
            frames = []
            valid_frame_detected = False  # 标记是否检测到有效帧
            start_frame = max(0, frame_num - self.sequence_length)  # 起始帧号，确保非负
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 设置视频读取位置

            prev_frame = None  # 用于插补缺失帧
            for _ in range(self.sequence_length):  # 读取 sequence_length 帧
                ret, frame = cap.read()
                if not ret:
                    logging.warning(f"视频帧读取失败，使用上一帧插补: {video_path}, 帧号: {start_frame}")
                    if prev_frame is not None:
                        frames.append(prev_frame)  # 插补缺失帧
                    continue
                
                # 使用 YOLOv8 检测人体
                results = list(self.yolo_model(frame, stream=True))  # Convert generator to list
                selected_bbox = None
                for idx, result in enumerate(results):
                    bbox = result.boxes.xyxy.cpu().numpy()  # 提取检测框 (x1, y1, x2, y2)
                    if (player == 'B' and idx == 0) or (player == 'A' and len(results) > 1 and idx == 1):
                        selected_bbox = bbox
                        break
                
                if selected_bbox is None:
                    logging.warning(f"未检测到人体，跳过帧: {video_path}, 帧号: {start_frame}")
                    if prev_frame is not None:
                        frames.append(prev_frame)  # 插补缺失帧
                    continue
                
                # 裁剪检测框内的图像
                x1, y1, x2, y2 = map(int, selected_bbox[0])  # 转换为整数
                cropped_frame = frame[y1:y2, x1:x2]  # 裁剪图像
                cropped_frame = cv2.resize(cropped_frame, resize_size)  # 动态调整大小
                cropped_frame = np.transpose(cropped_frame, (2, 0, 1))  # 转换为 (C, H, W)
                frames.append(cropped_frame)
                prev_frame = cropped_frame  # 更新上一帧
                valid_frame_detected = True  # 标记检测到有效帧

            cap.release()
            if len(frames) < self.sequence_length:
                raise ValueError(f"加载的帧数不足: {len(frames)}，预期: {self.sequence_length}")
            return np.array(frames), valid_frame_detected
        except Exception as e:
            logging.error(f"加载视频帧时出错: {e}, 视频路径: {video_path}, 帧号: {frame_num}")
            return None, False
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            sample = self.samples[idx]
            
            # Ensure required keys exist in the sample
            if 'video_path' not in sample or 'frame_num' not in sample or 'player' not in sample:
                raise KeyError(f"Sample is missing required keys: {sample}")
            
            # 加载视频帧
            frames, _ = self._load_frames(sample['video_path'], sample['frame_num'], sample['player'])
            if frames is None or len(frames) == 0:
                logging.warning(f"加载帧失败，返回空样本: {sample}")
                return torch.zeros((3, self.sequence_length, 224, 224)), torch.tensor(0)
                
            # 转换为tensor
            frames = torch.from_numpy(frames).float()  # (T, C, H, W)
            frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
            
            # 应用数据增强
            if self.transform:
                frames = self.transform(frames)
                
            return frames, torch.tensor(sample['shot_type'])
        except Exception as e:
            logging.error(f"获取样本时出错: {e}, 索引: {idx}")
            return torch.zeros((3, self.sequence_length, 224, 224)), torch.tensor(0)

def create_data_loaders(root_dir: str, video_dir: str, batch_size: int = 32, num_workers: int = 4,transform=None):
    """
    创建训练和验证数据加载器
    """
    # 创建数据集
    dataset = BadmintonDataset(root_dir, video_dir,transform=transform)
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(dataset)) # 80%训练集，20%验证集
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, # 
        shuffle=True, # 打乱数据集
        num_workers=num_workers, # 多线程加载数据
        pin_memory=True # 将数据加载到GPU内存中，加快训练速度
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
    # 测试数据加载
    root_dir = 'ShuttleSet'  # ShuttleSet目录
    video_dir = 'youtube_video'  # youtube_video目录
    train_loader, val_loader = create_data_loaders(root_dir, video_dir)
    
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    
    # 测试一个批次
    for frames, labels in train_loader:
        print(f"批次形状: {frames.shape}")
        print(f"标签形状: {labels.shape}")
        break