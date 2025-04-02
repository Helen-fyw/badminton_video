
---

# **Badminton Video Analysis System**

## **项目简介**
本项目是一个羽毛球比赛视频分析系统，旨在通过深度学习技术实现以下功能：
1. **人体姿态检测**：使用 YOLOv8-Pose 模型检测选手的姿态。
2. **羽毛球检测**：结合背景减法和 YOLOv5 模型检测羽毛球的位置。
3. **击球类型分类**：通过 3D 卷积神经网络（BadmintonShotNet）对击球类型进行分类。

---

## **功能特性**
- **视频处理**：使用 OpenCV 处理视频流，支持逐帧分析。
- **人体姿态检测**：基于 YOLOv8-Pose 模型，检测选手的关键点和骨架。
- **羽毛球检测**：结合背景减法和 YOLOv5 模型，检测羽毛球的轨迹和落点。
- **击球类型分类**：通过 3D 卷积网络对击球类型进行分类（如杀球、挑球等）。
- **可视化界面**：实时显示检测结果，包括选手位置、羽毛球轨迹和击球类型。

---

## **项目结构**
```
badminton_video/
├── data_preprocessing.py   # 数据预处理和加载器
├── model.py                # 模型定义
├── train.py                # 模型训练脚本
├── README.md               # 项目说明文档
├── ShuttleSet/             # 数据集目录
│   ├── match.csv           # 比赛信息
│   ├── homography.csv      # 单应矩阵信息
│   ├── set/                # 每场比赛的击球数据
├── youtube_video/          # 视频文件目录
│   ├── video1              # 比赛视频名称作为目录名
│       ├── video1.mp4      # 比赛视频
│   ├── video2              # 比赛视频名称作为目录名
│       ├── video2.mp4      # 比赛视频
├── video_ball_view_v2.py   # 训练视频可视化与动态捕捉
├── test.py                 # 测试视频可视化与动态捕捉
```

---

## **依赖项**
运行本项目需要以下依赖：
- Python 3.8+
- PyTorch 1.12+
- torchvision
- OpenCV
- NumPy
- pandas
- tqdm
- Ultralytics YOLO

### **安装依赖**
使用以下命令安装依赖：
```bash
pip install -r requirements.txt
```

---

## **使用方法**

### **1. 数据准备**
将比赛视频放置在 youtube_video 目录下，并确保 ShuttleSet 目录中包含以下文件：
- match.csv：比赛信息文件。
- `homography.csv`：单应矩阵信息文件。
- `set/`：每场比赛的击球数据（CSV 格式）。

### **2. 训练模型**
运行以下命令训练模型：
```bash
python train.py
```
- 默认使用 `batch_size=8` 和 `num_epochs=2`。
- 训练完成后，最佳模型会保存为 best_model.pth。

### **3. 测试模型**
运行以下命令测试模型：
```bash
python test.py
```
- 实时显示检测结果，包括选手位置、羽毛球轨迹和击球类型。
- 按 'q' 键退出。

### **4. 可视化结果**
运行以下命令查看训练过程中的可视化结果：
```bash
python video_ball_view_v2.py
```
- 实时显示检测结果，包括选手位置、羽毛球轨迹和击球类型。
- 按 'q' 键退出。

---

## **模型说明**

### **1. BadmintonShotNet**
- **输入**：`(batch_size, 3, 16, 224, 224)`，表示 16 帧 RGB 视频片段。
- **网络结构**：
  - 3 个 3D 卷积块，用于提取时空特征。
  - 全连接层，用于分类击球类型。
- **输出**：击球类型的概率分布。

### **2. YOLOv8-Pose**
- 用于检测选手的关键点和骨架。

### **3. YOLOv5**
- 用于检测羽毛球的位置。

---

## **日志记录**
训练和验证过程中的日志会保存到 `training_YYYYMMDD_HHMMSS.log` 文件中，包含以下信息：
- 每个 epoch 的训练损失和准确率。
- 每个 epoch 的验证损失和准确率。
- 最佳模型的保存记录。

---

## **注意事项**
1. **数据集路径**：
   - 确保 ShuttleSet 和 youtube_video 目录结构正确。
2. **显存限制**：
   - 如果显存不足，可以减小 `batch_size` 或使用梯度累积。
3. **模型参数**：
   - 可以在 train.py 中调整学习率（`lr`）和训练轮数（`num_epochs`）。

---

## **附录：`shot_type`**
`shot_type` 表示击球的类型，以下是支持的击球类型及其含义：
| 编号 | 类型名称       | 描述                     |
|------|----------------|--------------------------|
| 1    | Clear          | 高远球                   |
| 2    | Drop           | 吊球                     |
| 3    | Smash          | 杀球                     |
| 4    | Drive          | 平抽球                   |
| 5    | Net Shot       | 网前球                   |
| 6    | Lift           | 挑球                     |
| 7    | Push           | 推球                     |
| 8    | Block          | 挡网                     |
| 9    | Lob            | 挑高球                   |
| 10   | Net Kill       | 网前扑球                 |
| 11   | Net Lift       | 网前挑球                 |
| 12   | Cross Net      | 网前交叉球               |
| 13   | Cross Smash    | 斜线杀球                 |
| 14   | Cross Drop     | 斜线吊球                 |
| 15   | Cross Drive    | 斜线平抽球               |
| 16   | Cross Clear    | 斜线高远球               |
| 17   | Cross Push     | 斜线推球                 |
| 18   | Cross Block    | 斜线挡网                 |

---

## **参考文献**
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## **作者**
- 项目地址：[GitHub Repository](https://github.com/Helen-fyw/badminton_video)

---

