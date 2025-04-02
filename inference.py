import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import json
import os
from model import create_model

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BadmintonShotPredictor:
    def __init__(
        self,
        model_path: str,
        device: torch.device = None,
        sequence_length: int = 16,
        frame_size: Tuple[int, int] = (224, 224)
    ):
        """
        初始化预测器
        Args:
            model_path: 模型权重文件路径
            device: 运行设备
            sequence_length: 输入序列长度
            frame_size: 输入帧大小
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        
        # 加载模型
        self.model = create_model(num_classes=18, sequence_length=sequence_length, input_channels=3)
        self.model.load_state_dict(torch.load(model_path))  # ['model_state_dict']
        self.model.to(self.device)
        self.model.eval()
        
        # 击球类型映射
        self.shot_type_map = {
            0: '放小球', 1: '擋小球', 2: '殺球', 3: '点扣',
            4: '挑球', 5: '防守回挑', 6: '长球', 7: '平球',
            8: '小平球', 9: '后场抽平球', 10: '切球',
            11: '过渡切球', 12: '推球', 13: '扑球',
            14: '防守回抽', 15: '勾球', 16: '发短球', 17: '发长球'
        }
        
        # 视频文件映射
        self.video_mapping = {
            'Pusarla_V._Sindhu_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals': 
                'HSBC BWF World Tour Finals ｜ Day 3： Pornpawee Chochuwong (THA) vs. Pusarla V. Sindhu (IND) [Mawo3l3Hb9E].mp4',
            'Carolina_Marin_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_SemiFinals':
                'HSBC BWF World Tour Finals ｜ Day 4： Carolina Marin (ESP) [1] vs. Pornpawee Chochuwong (THA) [vfzkc3lFTdM].mp4',
            'Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals':
                'HSBC BWF World Tour Finals ｜ Day 5： Viktor Axelsen (DEN) vs. Anders Antonsen (DEN) [j7_cjmJDYNU].mp4',
            '282_1743336205.mp4':'282_1743336205.mp4'
        }
        
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理单帧图像"""
        # 调整大小
        frame = cv2.resize(frame, self.frame_size)
        # 归一化
        frame = frame.astype(np.float32) / 255.0
        # 转换为 (C, H, W) 格式
        frame = np.transpose(frame, (2, 0, 1))
        # 添加时间维度 (T, C, H, W)
        frame = np.expand_dims(frame, axis=0)
        return frame
    
    def process_video(self, video_path: str) -> List[Tuple[int, str, float]]:
        """
        处理视频并预测击球类型
        Args:
            video_path: 视频文件路径
        Returns:
            预测结果列表，每个元素为(帧号, 击球类型, 置信度)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = []
        frame_buffer = []
        
        with torch.no_grad():
            for frame_idx in tqdm(range(total_frames), desc="处理视频"):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 预处理帧
                processed_frame = self.preprocess_frame(frame)
                frame_buffer.append(processed_frame)
                
                # 当缓冲区达到指定长度时进行预测
                if len(frame_buffer) == self.sequence_length:
                    # 转换为tensor
                    input_tensor = torch.from_numpy(np.concatenate(frame_buffer, axis=0))  # (T, C, H, W)
                    input_tensor = input_tensor.to(self.device)
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (B, T, C, H, W)
                    
                    # 预测
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    # 记录结果
                    results.append((
                        frame_idx,
                        self.shot_type_map[predicted.item()],
                        confidence.item()
                    ))
                    
                    # 移除最旧的帧
                    frame_buffer.pop(0)
        
        cap.release()
        return results
    
    def save_results(self, results: List[Tuple[int, str, float]], output_path: str):
        """保存预测结果"""
        output_data = {
            'predictions': [
                {
                    'frame': frame_idx,
                    'shot_type': shot_type,
                    'confidence': confidence
                }
                for frame_idx, shot_type, confidence in results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"预测结果已保存到: {output_path}")
    
    def process_all_videos(self, video_dir: str, output_dir: str):
        """处理所有视频文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        for match_id, video_name in self.video_mapping.items():
            logging.info(f"处理视频: {video_name}")
            
            # 构建完整的视频路径
            video_path = os.path.join(video_dir, video_name)
            if not os.path.exists(video_path):
                logging.error(f"视频文件不存在: {video_path}")
                continue
            
            # 处理视频
            results = self.process_video(video_path)
            
            # 保存结果
            output_path = os.path.join(output_dir, f"{match_id}_predictions.json")
            self.save_results(results, output_path)
            
            # 打印统计信息
            shot_types = [r[1] for r in results]
            unique_shots = set(shot_types)
            logging.info(f"检测到的击球类型: {', '.join(unique_shots)}")
            logging.info(f"总预测帧数: {len(results)}")

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')
    
    # 创建预测器
    predictor = BadmintonShotPredictor(
        model_path='best_model.pth',  # 替换为您的模型路径
        device=device
    )
    
    # 设置目录
    video_dir = 'test_video'
    output_dir = 'prediction_results'
    
    # 处理所有视频
    predictor.process_all_videos(video_dir, output_dir)

if __name__ == '__main__':
    main()