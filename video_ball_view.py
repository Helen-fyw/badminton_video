import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from ultralytics import YOLO

# 加载预训练的姿态估计模型（YOLOv8n-pose 是轻量版）
model = YOLO("yolov8n-pose.pt")

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

# 创建背景减法器（用于羽毛球检测）
#TODO 1: 有关移动的参数设置
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=250, detectShadows=False)
#参数history表示用于建模的最新帧数，varT hreshold表示用于对比前景和背景的方差阈值，detectShadows表示是否检测阴影
#history和varThreshold的设置会影响背景建模的速度和准确性，detectShadows的设置会影响阴影检测的效果
#history: 用于建模的最新帧数，表示背景建模的历史帧数，即用于建模的最新帧数，值越大，背景建模的效果越好，但速度越慢
#varThreshold: 在检测物体移动速度很快时，应当适当增大varThreshold，否则可能会导致背景建模不准确

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

# 提取label中有效时间的区间
frame_nums=pd.merge(labels.groupby('rally')['frame_num'].max().reset_index(drop=False),labels.groupby('rally')['frame_num'].min().reset_index(drop=False),on='rally')
frame_nums.columns=['rally','max_frame_num','min_frame_num']
frame_nums['max_frame_num']=frame_nums['max_frame_num'].astype(int)
frame_nums['min_frame_num']=frame_nums['min_frame_num'].astype(int)

# 遍历视频帧
rally_num = 0  # 当前回合编号
frame_index = frame_nums['min_frame_num'][0]
paused = False  # 播放状态标志

while cap.isOpened():  # 检查视频是否打开成功
    if not paused:  # 如果未暂停，则读取下一帧
        ret, frame = cap.read()  # 读取视频帧
        frame_copy = frame.copy()  # 复制帧用于绘制
        if not ret:  # 如果读取失败，则退出循环
            break
        spoted_flag = False  # 是否检测到合格帧
        # 使用 YOLOv8-Pose 进行人体姿态推理
        results = model(frame, stream=True)
        # results_bone = list(results).copy()  # 复制结果以便后续处理
        results_bone=[]
        test=0
        for result in results:
            test+=1
            print(test)
            results_bone.append(result.keypoints.xyn.clone())
            frame = result.plot(kpt_radius=1, line_width=1, font_size=0.2) 
        #显示results_bone的shape
        # print(results_bone[0].shape)


        #对于每一行数据，判断当前帧是否在有效时间区间内，如果是则执行for循环中的代码
        for _, row in frame_nums.iterrows():
            if (frame_index >= row['min_frame_num']) & (frame_index <= row['max_frame_num']+50):
                spoted_flag = True
                # 使用背景减法器检测羽毛球
                gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                fgmask = fgbg.apply(gray)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 5*5的椭圆形结构元素，越大检测到的物体越大
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

                # 检测轮廓
                contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_num=0
                for contour in contours:

                    # TODO 2: 有关羽毛球面积长宽的参数
                    if (cv2.contourArea(contour) < 10) & (cv2.contourArea(contour) > 100):  # 忽略小面积的轮廓
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    if w < 8 and h < 8:  # 假设羽毛球是一个小的快速移动物体
                        # 检查是否在人体姿态检测区域内
                        is_in_pose_area = False
                        for result in results_bone[0]:
                            for keypoint in result:  # 获取关键点坐标
                                kx, ky = keypoint
                                kx= kx*width
                                ky= ky*height
                                if kx - 20 <= x <= kx + 20 and ky-20 <= y <= ky+20:
                                    is_in_pose_area = True
                                    break
                            if is_in_pose_area:
                                break

                        if is_in_pose_area:
                            continue  # 如果在人体姿态检测区域内，则跳过标记

                        # 提取羽毛球区域的颜色
                        shuttlecock_region = gray[y:y + h, x:x + w]
                        mean_color = cv2.mean(shuttlecock_region)[0]  # 获取灰度值的平均值

                        # 判断颜色是否为浅灰或白色
                        # TODO 3: 有关颜色的参数设置
                        if (120 <= mean_color <= 230) :  # 浅灰或白色的灰度值范围
                            if contour_num == 0:
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame, "Shuttlecock", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                contour_num += 1

                        # TODO 一些别的设置，比如在人像框里的羽毛球不显示

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
                player_x, player_y = row["player_location_x"] / 2, row["player_location_y"] / 2
                draw.rectangle(
                    [(player_x - 10, player_y - 10), (player_x + 10, player_y + 10)],
                    outline="white",
                    width=3
                )

                # 在指定坐标绘制红色方框（对手位置）
                opponent_x, opponent_y = row["opponent_location_x"] / 2, row["opponent_location_y"] / 2
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
        if spoted_flag :
            cv2.imshow("Video", frame)
            # 更新滑动条位置
            cv2.setTrackbarPos("Progress", "Video", frame_index)
            frame_index += 1
        else:
            rally_num += 1
            frame_index = frame_nums['min_frame_num'][rally_num]


    # 按下 'q' 键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # 按下空格键暂停/播放
        paused = not paused

# 释放资源
cap.release()
cv2.destroyAllWindows()