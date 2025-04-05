"""
羽毛球视频分析系统

本系统用于分析羽毛球比赛视频，实现以下功能：
1. 人体姿态检测：使用YOLOv8-pose模型检测选手姿态
2. 羽毛球检测：使用背景减法和颜色特征检测羽毛球位置

主要组件：
- 视频处理：使用OpenCV处理视频流
- 姿态检测：使用YOLOv8-pose模型
- 羽毛球检测：使用背景减法(MOG2)和颜色特征分析
- 界面显示：使用OpenCV实现可视化

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
- NumPy
- Ultralytics YOLO
"""

import cv2
import numpy as np
from ultralytics import YOLO

# 初始化YOLOv8姿态估计模型
# 使用轻量级版本yolov8n-pose以提高处理速度
model = YOLO("yolov8n-pose.pt")

# 设置视频文件路径
video_path = "466_1743438943_black.mp4"

# 初始化视频捕获对象
cap = cv2.VideoCapture(video_path)

# 获取视频基本信息
fps = int(cap.get(cv2.CAP_PROP_FPS))  # 帧率
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 视频宽度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 视频高度
print(width, height)

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

# 初始化视频播放相关变量
frame_index = 72 # 当前帧索引
paused = False  # 播放状态标志

# 主循环：处理视频帧
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        frame_copy = frame.copy()
        if not ret:
            break
            
        # 使用YOLOv8进行人体姿态检测
        results = model(frame, stream=True)
        results_bone = []
        test = 0
        for result in results:
            test += 1
            print(test)
            results_bone.append(result.keypoints.xyn.clone())
            frame = result.plot(kpt_radius=1, line_width=1, font_size=0.2)
                
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
            if (cv2.contourArea(contour) < 10*(3**2)) & (cv2.contourArea(contour) > 110*(3**2) ):
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10*5 and h < 10*5:  # 羽毛球尺寸阈值
                # 检查是否在人体姿态检测区域内
                is_in_pose_area = False
                for result in results_bone[0]:
                    for keypoint in result:
                        kx, ky = keypoint
                        kx = kx*width
                        ky = ky*height
                        if kx - 20*3 <= x <= kx + 20*3 and ky-20*3 <= y <= ky+20*3:
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

        # 显示处理后的帧
        cv2.imshow("Video", frame)
        cv2.setTrackbarPos("Progress", "Video", frame_index)
        frame_index += 1

    # 键盘控制
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 按q退出
        break
    elif key == ord(' '):  # 按空格暂停/继续
        paused = not paused

# 释放资源
cap.release()
cv2.destroyAllWindows()