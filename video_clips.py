#pip install opencv-python
import cv2

# 打开视频文件
video_path=r'E:\PythonLearning\Statistic_Project\37 TOYOTA Thailand Open ｜ Day 6： Hans-Kristian Solberg VIittinghus (DEN) vs. Viktor Axelsen (DEN) [4] [4rQUHv9oGpI].mp4'
#cap = cv2.VideoCapture(video_path)

# 已知起点帧、终点帧，完成视频切片
# 设置输入视频路径和输出视频路径
input_video_path = video_path
output_video_path = r'E:\PythonLearning\Statistic_Project\output_video.mp4'

# 创建 VideoCapture 对象
cap = cv2.VideoCapture(input_video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    # 获取视频的宽度和高度
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 定义编码器并创建 VideoWriter 对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 初始化帧计数器
    frame_count = 0

    # 循环读取每一帧
    while True:
        ret, frame = cap.read()

        # 如果无法读取帧，退出循环
        if not ret:
            break

        # 检查当前帧是否在指定范围内
        # If循环 在【指定帧数范围内】抽取帧合成视频。这里原视频1秒30帧，300帧就是10秒的切片。
        if 1000 <= frame_count <= 1300:
            # 写入帧到输出视频
            out.write(frame)
        elif frame_count > 1300:
            # 如果已经处理完所需的帧范围，退出循环
            break

        # 增加帧计数器
        frame_count += 1

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved to {output_video_path}")



