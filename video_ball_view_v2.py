"""
羽毛球视频分析系统

本系统用于分析羽毛球比赛视频，实现以下功能：
1. 人体姿态检测：使用YOLOv8-pose模型检测选手姿态
2. 羽毛球检测：使用背景减法和颜色特征检测羽毛球位置
3. 动作标注：显示选手动作类型、位置和击球点

主要组件：
- 视频处理：使用OpenCV处理视频流
- 姿态检测：使用YOLOv8-pose模型
- 羽毛球检测：使用背景减法(MOG2)和颜色特征分析
- 界面显示：使用OpenCV和PIL实现可视化

使用方法：
1. 运行程序后会显示视频窗口
2. 使用空格键暂停/继续播放
3. 使用进度条控制视频播放位置
4. 按q键退出程序

待优化项：
1. 背景减法器参数优化
2. 羽毛球检测参数优化
3. 羽毛球颜色特征优化

依赖项：
- OpenCV (cv2)
- Pandas
- PIL (Pillow)
- NumPy
- Ultralytics YOLO
"""



import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# 初始化YOLOv8姿态估计模型
# 使用轻量级版本yolov8n-pose以提高处理速度
model = YOLO("yolov8n-pose.pt")

# 设置视频和标签文件路径
video_path = "youtube_video/Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals/Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals.mp4"
label_path = "ShuttleSet/set/Anders_Antonsen_Viktor_Axelsen_HSBC_BWF_WORLD_TOUR_FINALS_2020_Finals/set3.csv"

# 读取CSV标签文件
labels = pd.read_csv(label_path)

# 初始化视频捕获对象
cap = cv2.VideoCapture(video_path)

# 获取视频基本信息
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度

# TODO 1: 背景减法器参数优化
# 初始化背景减法器用于羽毛球检测
# 参数说明：
# history: 用于背景建模的历史帧数，值越大建模效果越好但计算量越大
# varThreshold: 前景检测的方差阈值，值越大对运动越敏感
# detectShadows: 是否检测阴影，关闭可提高性能
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=250, detectShadows=False)

# 创建可调整大小的视频窗口
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# 定义进度条回调函数
def on_trackbar(val):
    global frame_index
    frame_index = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

# 创建视频进度控制条
cv2.createTrackbar("Progress", "Video", 0, total_frames - 1, on_trackbar)

# 加载中文字体
font_path = "SimHei.ttf"  # 请确保字体文件存在
font = ImageFont.truetype(font_path, 32)

# 提取每个回合的有效帧区间
frame_nums = pd.merge(
    labels.groupby('rally')['frame_num'].max().reset_index(drop=False),
    labels.groupby('rally')['frame_num'].min().reset_index(drop=False),
    on='rally'
)
frame_nums.columns = ['rally', 'max_frame_num', 'min_frame_num']
frame_nums['max_frame_num'] = frame_nums['max_frame_num'].astype(int)
frame_nums['min_frame_num'] = frame_nums['min_frame_num'].astype(int)

# 初始化视频播放相关变量
rally_num = 0  # 当前回合编号
frame_index = frame_nums['min_frame_num'][0]  # 当前帧索引
paused = False  # 播放状态标志

# 主循环：处理视频帧
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        frame_copy = frame.copy()
        if not ret:
            break
            
        spoted_flag = False  # 标记是否检测到有效帧
        
        # 使用YOLOv8进行人体姿态检测
        results = model(frame, stream=True)
        results_bone = []
        test = 0
        for result in results:
            test += 1
            print(test)
            results_bone.append(result.keypoints.xyn.clone())
            frame = result.plot(kpt_radius=1, line_width=1, font_size=0.2)

        # 处理每个回合的有效帧区间
        for _, row in frame_nums.iterrows():
            if (frame_index >= row['min_frame_num']) & (frame_index <= row['max_frame_num']+50):
                spoted_flag = True
                
                # 使用背景减法检测羽毛球
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                fgmask = fgbg.apply(gray)
                
                # 使用形态学操作去除噪点
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                # 检测并筛选轮廓
                contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_num = 0
                
                for contour in contours:
                    # TODO 2: 羽毛球检测参数优化
                    # 根据面积筛选可能的羽毛球区域
                    # 面积范围需要根据实际情况调整
                    if (cv2.contourArea(contour) < 10) & (cv2.contourArea(contour) > 100):
                        continue
                        
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < 8 and h < 8:  # 羽毛球尺寸阈值
                        # 检查是否在人体姿态检测区域内
                        is_in_pose_area = False
                        for result in results_bone[0]:
                            for keypoint in result:
                                kx, ky = keypoint
                                kx = kx*width
                                ky = ky*height
                                if kx - 20 <= x <= kx + 20 and ky-20 <= y <= ky+20:
                                    is_in_pose_area = True
                                    break
                            if is_in_pose_area:
                                break

                        if is_in_pose_area:
                            continue  # 跳过人体区域内的检测结果

                        # TODO 3: 羽毛球颜色特征优化
                        # 分析羽毛球区域的颜色特征
                        shuttlecock_region = gray[y:y + h, x:x + w]
                        mean_color = cv2.mean(shuttlecock_region)[0]

                        # 根据颜色特征判断是否为羽毛球
                        # 颜色范围需要根据实际羽毛球颜色调整
                        if (120 <= mean_color <= 230):
                            if contour_num == 0:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, "Shuttlecock", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                contour_num += 1

        # 绘制标签信息
        for _, row in labels.iterrows():
            if abs(row['frame_num'] - frame_index) < 5:
                text = row['type']

                # 转换图像格式用于绘制中文
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)

                # 绘制动作类型文字
                draw.text((50, 50), text, font=font, fill=(0, 255, 0, 255))

                # 绘制选手位置标记
                player_x, player_y = row["player_location_x"] / 2, row["player_location_y"] / 2
                draw.rectangle(
                    [(player_x - 10, player_y - 10), (player_x + 10, player_y + 10)],
                    outline="white",
                    width=3
                )

                # 绘制对手位置标记
                opponent_x, opponent_y = row["opponent_location_x"] / 2, row["opponent_location_y"] / 2
                draw.rectangle(
                    [(opponent_x - 10, opponent_y - 10), (opponent_x + 10, opponent_y + 10)],
                    outline="red",
                    width=3
                )

                # 绘制击球位置标记
                hit_x, hit_y = row["hit_x"] / 2, row["hit_y"] / 2
                draw.ellipse(
                    [(hit_x - 10, hit_y - 10), (hit_x + 10, hit_y + 10)],
                    outline="yellow",
                    width=3
                )

                # 转换回OpenCV格式
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        # 显示处理后的帧
        if spoted_flag:
            cv2.imshow("Video", frame)
            cv2.setTrackbarPos("Progress", "Video", frame_index)
            frame_index += 1
        else:
            rally_num += 1
            frame_index = frame_nums['min_frame_num'][rally_num]

    # 键盘控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按q退出
        break
    elif key == ord(' '):  # 按空格暂停/继续
        paused = not paused

# 释放资源
cap.release()
cv2.destroyAllWindows()