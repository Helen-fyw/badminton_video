import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import math
from colorama import Fore, Style  # Import colorama for colored output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class BadmintonShotNet(nn.Module):
    def __init__(self, num_classes=18, sequence_length=16, input_size=(224, 224)):
        super(BadmintonShotNet, self).__init__()
        
        # 输入尺寸
        self.input_size = input_size  # (height, width)
        
        # 3D卷积层 - 使用步长卷积替代池化
        # 输入: (batch_size, 3, sequence_length, height, width)
        # 3D卷积块1
        self.conv3d_1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_1 = nn.BatchNorm3d(8)
        self.conv3d_1_stride = nn.Conv3d(8, 8, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 3D卷积块2
        self.conv3d_2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_2 = nn.BatchNorm3d(16)
        self.conv3d_2_stride = nn.Conv3d(16, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        # 3D卷积块3
        self.conv3d_3 = nn.Conv3d(16, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3d_3 = nn.BatchNorm3d(32)
        self.conv3d_3_stride = nn.Conv3d(32, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        # 动态计算展平后的特征维度
        self.flat_features = self._calculate_flat_features(sequence_length)
        print(f"{Fore.YELLOW}展平后的特征维度: {self.flat_features}{Style.RESET_ALL}")
        
        # 全连接层
        self.fc1 = nn.Linear(self.flat_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
    
    def _calculate_flat_features(self, sequence_length):
        """动态计算展平后的特征维度"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, sequence_length, *self.input_size)
            x = self.conv3d_1(dummy_input)
            x = self.bn3d_1(x)
            x = F.relu(x)
            x = self.conv3d_1_stride(x)
            x = self.conv3d_2(x)
            x = self.bn3d_2(x)
            x = F.relu(x)
            x = self.conv3d_2_stride(x)
            x = self.conv3d_3(x)
            x = self.bn3d_3(x)
            x = F.relu(x)
            x = self.conv3d_3_stride(x)
            return x.numel()

    def forward(self, x):
        # Remove or comment out print statements for better performance
        print(f"输入形状: {x.shape}")
        x = self.conv3d_1(x)
        x = self.bn3d_1(x)
        x = F.relu(x)
        x = self.conv3d_1_stride(x)
        print(f"卷积块1后形状: {x.shape}")
        x = self.conv3d_2(x)
        x = self.bn3d_2(x)
        x = F.relu(x)
        x = self.conv3d_2_stride(x)
        print(f"卷积块2后形状: {x.shape}")
        x = self.conv3d_3(x)
        x = self.bn3d_3(x)
        x = F.relu(x)
        x = self.conv3d_3_stride(x)
        print(f"卷积块3后形状: {x.shape}")
        x = x.view(x.size(0), -1)
        print(f"展平后形状: {x.shape}")
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Convert model to TorchScript for runtime optimization
def create_model(num_classes: int = 18, sequence_length: int = 16, input_size=(224, 224)) -> BadmintonShotNet:
    """
    创建模型实例
    Args:
        num_classes: 击球类型数量
        sequence_length: 输入视频帧数
        input_size: 输入图像的尺寸 (height, width)
    Returns:
        模型实例
    """
    model = BadmintonShotNet(num_classes=num_classes, sequence_length=sequence_length, input_size=input_size)
    return torch.jit.script(model)  # Convert to TorchScript

if __name__ == '__main__':
    # 测试模型
    model = create_model(input_size=(100, 50))  # 测试动态输入尺寸
    x = torch.randn(32, 3, 16, 100, 50)  # 批次大小为32，3个通道，16帧，100x50分辨率
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")