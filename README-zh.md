# FitnessPose

**FitnessPose** 是一个基于 **YOLOv8-Pose 模型** 的自动化姿态分析工具，专注于对俯卧撑动作进行精准的姿态识别、标准度评估以及动作不平衡性检测。用户可通过运行 `app.py` 脚本传入视频文件（或使用摄像头实时分析），系统将逐帧解析动作，输出详细的评估指标及可视化结果。

## 开始

### python

建议使用python虚拟环境，可以使用anaconda进行一键部署

```bash
# 克隆项目到本地
git clone https://github.com/flow111817/FitnessPose.git

# 进入项目目录
cd FitnessPose

# 创建并激活虚拟环境（推荐使用 conda）
conda create -n fitness_pose python=3.8
conda activate fitness_pose

# 安装依赖包
pip install -r requirements.txt

# 启动应用（支持视频文件或摄像头输入）
python app.py
```

- 如果你没有安装 Conda，可以前往 Anaconda官网 或使用 Miniconda 轻量部署。
- 本项目依赖 OpenCV、YOLOv8（来自 Ultralytics）、numpy 等库，使用虚拟环境可以有效避免版本冲突。
- 本项目使用的pytorch版本为2.4.1，cudatoolkit推荐下载11.8版本，即可对模型使用gpu进行微调

### Docker

```bash
# 拉取镜像
docker pull flow0817817/fitnesspose:latest

# 启动
docker run -d -p 5000:5000 --name test flow0817817/fitnesspose:latest
```

## 贡献

欢迎并鼓励为**FitnessPose**做出贡献。
