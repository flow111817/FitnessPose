# 模型路径配置
MODEL_PATH = "models/yolov8s_finetuned.pt"

# 关键点索引配置
KEYPOINT_INDICES = {
    'NOSE': 0,
    'LEFT_EYE': 1,
    'RIGHT_EYE': 2,
    'LEFT_EAR': 3,
    'RIGHT_EAR': 4,
    'LEFT_SHOULDER': 5,
    'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7,
    'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9,
    'RIGHT_WRIST': 10,
    'LEFT_HIP': 11,
    'RIGHT_HIP': 12,
    'LEFT_KNEE': 13,
    'RIGHT_KNEE': 14,
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16
}

# 帧间异常值过滤容忍值
MAX_ABNORMAL_FRAMES = 3

# 帧间最大跳动阈值
MAX_DELTA=150

# 指数加权滑动平均指数（值越高平滑越高，延迟越高）
ALPHA = 0.5

#相机启动索引
CAMERA_INDEX = 0

# 报告生成地址
REPORT_PATH = "./data/reports/"
VIDEO_PATH = "./data/videos/"
