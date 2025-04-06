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
from multiprocessing import cpu_count  # Import cpu_count for dynamic num_workers
from torch.cuda.amp import GradScaler, autocast  # Import for mixed precision training

from data_preprocessing import create_data_loaders
from model import BadmintonShotNet

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', mode='a', encoding='utf-8'),  # 追加模式
        logging.StreamHandler()
    ],
    level=logging.INFO  # 设置日志级别
)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, min_lr=1e-6)
    scaler = GradScaler()

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for batch_idx, (frames, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training')):
            try:
                frames, labels = frames.to(device), labels.to(device)
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            except Exception as e:
                logging.error(f"Error in training loop at batch {batch_idx}: {e}")
                continue

        train_acc = 100. * train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)  # Precompute average loss

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.inference_mode():
            for frames, labels in val_loader:
                try:
                    frames, labels = frames.to(device), labels.to(device)
                    outputs = model(frames)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                except Exception as e:
                    logging.error(f"Error in validation loop: {e}")
                    continue
        
        val_acc = 100. * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)  # Precompute average loss

        scheduler.step(val_loss_avg)

        # Logging and saving the best model
        logging.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        print(f"{Fore.GREEN}Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%{Style.RESET_ALL}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info(f"Saved best model with Val Acc: {best_val_acc:.2f}%")

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置设备
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner for consistent input sizes
    else:
        device = torch.device("cpu")
    logging.info(f'使用设备: {device}')
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((50, 100)),
        transforms.ToTensor(),  # Ensure tensors are created
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 动态设置批次大小和子进程数量
    batch_size = 16 if device.type == "mps" else 32  # Reduce batch size for MPS
    num_workers = 6  
    
    logging.info(f'批次大小: {batch_size}, 子进程数量: {num_workers}')
    
    # 创建数据加载器
    root_dir = 'ShuttleSet/set'
    video_dir = 'youtube_video'
    
    train_loader, val_loader = create_data_loaders(
        root_dir, 
        video_dir, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        transform=transform  # 确保传递transform
    )
    
    # 创建模型
    model = BadmintonShotNet(num_classes=18).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练模型
    num_epochs = 3
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

if __name__ == '__main__':
    main()