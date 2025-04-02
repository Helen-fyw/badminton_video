import cv2
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ultralytics import YOLO

# 加载预训练的姿态估计模型（YOLOv8n-pose 是轻量版，YOLOv8x-pose 是高精度版）
model = YOLO("yolov8n-pose.pt")  # 也可以选择 yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose

# 读取视频和标签文件
video_path = "youtube_video/HSBC BWF World Tour Finals ｜ Day 3： Pornpawee Chochuwong (THA) vs. Pusarla V. Sindhu (IND) [Mawo3l3Hb9E].mp4"
label_path = "ShuttleSet/set/Pusarla_V._Sindhu_Pornpawee_Chochuwong_HSBC_BWF_WORLD_TOUR_FINALS_2020_QuarterFinals/set1.csv"

# 加载CSV文件
labels = pd.read_csv(label_path)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率和总帧数
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频窗口
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# 定义滑动条的回调函数
def on_trackbar(val):
    global frame_index
    frame_index = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

# 创建滑动条
cv2.createTrackbar("Progress", "Video", 0, total_frames - 1, on_trackbar)

# 加载字体（确保路径正确，使用支持中文的字体文件）
font_path = "SimHei.ttf"  # 替换为本地的中文字体路径
font = ImageFont.truetype(font_path, 32)

# 遍历视频帧
frame_index = 0
paused = False  # 播放状态标志

while cap.isOpened(): # 检查视频是否打开成功
    if not paused: # 如果未暂停，则读取下一帧
        ret, frame = cap.read() # 读取视频帧
        if not ret: # 如果读取失败，则退出循环
            break

        # 使用 YOLOv8-Pose 进行推理
        results = model(frame, stream=True)  # stream=True 适用于视频流

        for result in results:
            # 绘制检测框和关键点
            frame = result.plot(
                kpt_radius=1,      # 关键点半径调小
                line_width=1,       # 检测框线条调细
                font_size=0.2       # 字体调小
            )  # 自动绘制检测框和关键点

            # # 显示关键点坐标（可选）
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()  # 获取关键点坐标 (N, 17, 2)
                for kpts in keypoints:
                    print("检测到的人体关键点坐标：", kpts)  # 17个COCO格式关键点

        # # 查找当前时间的标签
        for _, row in labels.iterrows():
            if abs(row['frame_num'] - frame_index) < 5:  # 时间匹配
                text = row['type']

                # 将 OpenCV 图像转换为 PIL 图像
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                
                # 在帧上绘制中文文字
                draw.text((50, 50), text, font=font, fill=(0, 255, 0, 255))
                
                # 在指定坐标绘制白色方框（玩家位置）
                player_x, player_y = row["player_location_x"]/2, row["player_location_y"]/2
                draw.rectangle(
                    [(player_x - 10, player_y - 10), (player_x + 10, player_y + 10)],
                    outline="white",
                    width=3
                )
                
                
                # 在指定坐标绘制红色方框（对手位置）
                opponent_x, opponent_y = row["opponent_location_x"]/2, row["opponent_location_y"]/2
                draw.rectangle(
                    [(opponent_x - 10, opponent_y - 10), (opponent_x + 10, opponent_y + 10)],
                    outline="red",
                    width=3
                )

                # 在指定坐标绘制黄色圆圈（击球位置）
                hit_x, hit_y = row["hit_x"] / 2, row["hit_y"] / 2
                draw.ellipse(
                    [(hit_x - 10, hit_y - 10), (hit_x + 10, hit_y + 10)],
                    outline="yellow",
                    width=3
                )

                # 将 PIL 图像转换回 OpenCV 图像
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 显示视频帧
        cv2.imshow("Video", frame) #是显示视频帧的函数，用于在窗口中显示视频帧。

        # 更新滑动条位置
        cv2.setTrackbarPos("Progress", "Video", frame_index) #是设置滑动条的位置，用于更新滑动条的位置。

        frame_index += 1

    # 按下 'q' 键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # 按下空格键暂停/播放
        paused = not paused



# 释放资源
cap.release()
cv2.destroyAllWindows()