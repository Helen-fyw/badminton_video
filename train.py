import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple
import logging
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision import transforms
from colorama import Fore, Style  # Import colorama for colored output
import multiprocessing
multiprocessing.set_start_method('fork', force=True)  # Use 'fork' start method

from data_preprocessing import create_data_loaders
from model import BadmintonShotNet

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (frames, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            try:
                # 这里的frames是一个batch_size * 3 * 16 * 224 * 224的tensor
                frames = frames.to(device)
                # labels是一个batch_size * 1的tensor
                labels = labels.to(device)
                
                optimizer.zero_grad()  # 清空梯度
                # 前向传播
                outputs = model(frames)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()  # 计算正确预测的数量
            except Exception as e:
                logging.error(f"{Fore.WHITE}Error in training loop at batch {batch_idx}: {e}{Style.RESET_ALL}")
                continue  # 跳过当前batch

        train_acc = 100. * train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for i, (frames, labels) in enumerate(val_loader):
                try:
                    print(f'\r{Fore.WHITE}Epoch {epoch+1}/{num_epochs} - Validation: {i+1}/{len(val_loader)} batches{Style.RESET_ALL}', end='')
                    frames = frames.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                except Exception as e:
                    logging.error(f"{Fore.WHITE}Error in validation loop at batch {i}: {e}{Style.RESET_ALL}")
                    continue  # 跳过当前batch
        
        val_acc = 100. * val_correct / val_total
        
        # 打印训练信息
        logging.info(f"{Fore.WHITE}Epoch {epoch+1}/{num_epochs}:{Style.RESET_ALL}")
        logging.info(f"{Fore.YELLOW}Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%{Style.RESET_ALL}")
        logging.info(f"{Fore.YELLOW}Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%{Style.RESET_ALL}")
        
        print(f"{Fore.WHITE}Epoch {epoch+1}/{num_epochs}:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%{Style.RESET_ALL}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"{Fore.YELLOW}保存最佳模型，验证准确率: {best_val_acc:.2f}%{Style.RESET_ALL}")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # 设置MPS内存管理
        # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'  # 设置内存使用上限为80%
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f'使用设备: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # mean和std值根据数据集进行调整，这里使用ImageNet的均值和标准差，可以根据实际情况调整
    ])
    
    # 创建数据加载器
    root_dir = 'ShuttleSet/set'
    video_dir = 'youtube_video'
    # 减小批次大小，batch_size指定每个批次的样本数量，num_workers指定加载数据时使用的子进程数量
    train_loader, val_loader = create_data_loaders(root_dir, video_dir, batch_size=8, num_workers=2)
    
    # 创建模型
    # TODO 是一共10类吧
    model = BadmintonShotNet(num_classes=18).to(device) #num_classes表示分类的类别数，这里假设有10个类别
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # TODO 学习率
    
    # 训练模型
    num_epochs = 1
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()