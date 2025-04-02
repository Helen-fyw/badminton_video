# pip install OpenCV-python
import cv2
import pandas as pd
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
# 读取视频和标签文件
#.表示当前目录，..表示上一级目录
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
frame_index = 3000
paused = False  # 播放状态标志

while cap.isOpened():
    if not paused: # 如果未暂停，则读取下一帧
        ret, frame = cap.read()
        if not ret:
            break

        # 查找当前时间的标签
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
        cv2.imshow("Video", frame)

        # 更新滑动条位置
        cv2.setTrackbarPos("Progress", "Video", frame_index)

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