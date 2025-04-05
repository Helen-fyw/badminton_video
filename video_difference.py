#参考图片为经过PS得到的球场背景，把视频和参考图片做 “差值”（Difference）混合模式。“差值” 模式会比较上下两个图层的像素颜色，相同颜色区域显示为黑色（即差值为 0），不同颜色区域则显示出颜色差异。
#目的是想对原视频尽可能地减少冗余信息，使后续识别能更准确

import cv2

reference_image_path = r'..\Statistic_Project\ShuttleSet\data_37\Court_reference.png'
input_video_path=r'..\Statistic_Project\ShuttleSet\data_37\TOYOTA Thailand Open ｜ Day 6： Hans-Kristian Solberg VIittinghus (DEN) vs. Viktor Axelsen (DEN) [4] [4rQUHv9oGpI].mp4'

# 读取参考图片、待处理的视频
reference_image = cv2.imread(reference_image_path)
if reference_image is None:
    print(f"Error: Could not read the reference image from {reference_image_path}")
    exit(1)

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open the video file from {input_video_path}")
    exit(1)
# # 创建视频窗口
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# Get the total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Define the callback function for the slider
def on_trackbar(val):
    global frame_index
    frame_index = val
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
# Create a slider
cv2.createTrackbar("Progress", "Video", 0, total_frames - 1, on_trackbar)

# 对视频做差分
threshold = 50  # 设置阈值

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 确保参考图片和视频帧的尺寸一致
    frame = cv2.resize(frame, (reference_image.shape[1], reference_image.shape[0]))

    # 计算差值
    difference = cv2.absdiff(frame, reference_image)

    # 创建一个掩码，标记差异较大的像素
    mask = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    _, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)  # 应用阈值

    # 将掩码应用到原始帧
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 将掩码转换为三通道
    result = cv2.bitwise_and(frame, mask)  # 保留差异较大的像素
    result += cv2.bitwise_and(difference, cv2.bitwise_not(mask))  # 将差异较小的像素设置为黑色

    # 显示结果
    cv2.imshow('Difference', result)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

