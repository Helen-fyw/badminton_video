import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import json

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
        
        # 击球类型映射
        self.shot_type_map = {'發短球': 1, '長球': 2, '推球': 3, '殺球':4, '擋小球':5, '平球':6, '放小球':7, '挑球':8, '切球':9, '發長球':10, '接不到':11} 

        print("\n数据集中的比赛：")
        for _, match in self.matches.iterrows():
            video_path = os.path.join(self.video_dir,match['video'], match['video'] + '.mp4')
            if os.path.exists(video_path):
                print(f"✓ {match['video']}")
            else:
                print(f"✗ {match['video']}")
        
        # 收集所有样本
        self.samples = self._collect_samples()
        
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
                    # TODO 这里似乎不用判断i < self.sequence_length
                    # if i < self.sequence_length:
                    #     continue
                        
                    # 获取击球类型
                    shot_type = set_data.iloc[i]['type']
                    if shot_type not in self.shot_type_map:
                        continue
                        
                    # 获取帧号
                    frame_num = set_data.iloc[i]['frame_num']
                    
                    # 构建样本
                    sample = {
                        'video_path': video_path,
                        'frame_num': frame_num,
                        'shot_type': self.shot_type_map[shot_type],
                        'sequence_length': self.sequence_length
                    }
                    samples.append(sample)
                    
        if not samples:
            raise ValueError("没有找到有效的样本！请检查数据集路径和视频文件是否正确。")
                    
        return samples
    
    def _load_frames(self, video_path: str, frame_num: int) -> np.ndarray:
        """加载视频帧"""
        cap = cv2.VideoCapture(video_path)
        
        frames = []
        start_frame = frame_num - self.sequence_length
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(self.sequence_length):
            ret, frame = cap.read()
            if not ret:
                return None
            # 调整大小
            frame = cv2.resize(frame, (224, 224))
            # 转换为 (C, H, W) 格式
            frame = np.transpose(frame, (2, 0, 1))
            # 添加到帧列表
            frames.append(frame)
            
        cap.release()
        return np.array(frames)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        
        # 加载视频帧
        frames = self._load_frames(sample['video_path'], sample['frame_num'])
        if frames is None:
            # 如果加载失败，返回一个空的样本
            return torch.zeros((3, self.sequence_length, 224, 224)), torch.tensor(0)
            
        # 转换为tensor
        frames = torch.from_numpy(frames).float()  # (T, C, H, W)
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        
        # 应用数据增强
        if self.transform:
            frames = self.transform(frames)
            
        return frames, torch.tensor(sample['shot_type'])

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