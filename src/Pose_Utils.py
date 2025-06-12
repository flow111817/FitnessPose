import numpy as np
from config import KEYPOINT_INDICES

def calculate_angle(a, b, c):
    """计算三点之间的角度，处理无效点"""
    # 检查点是否有效
    if a is None or b is None or c is None:
        return 180  # 返回默认值
    
    a, b, c = np.array(a), np.array(b), np.array(c)
    
    # 检查点是否为零点
    if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6 or np.linalg.norm(c) < 1e-6:
        return 180  # 返回默认值
    
    ba, bc = a - b, c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    # 避免除以零
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 180  # 返回默认值
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cosine_angle))

def is_kp_valid(keypoints, name):
    """检查关键点是否存在且置信度足够"""
    idx = KEYPOINT_INDICES.get(name, -1)
    if idx < 0 or idx >= len(keypoints):
        return False
    
    kp = keypoints[idx]
    # 检查关键点坐标是否有效 (非零且置信度足够)
    if len(kp) < 3 or kp[2] < 0.1:  # 置信度过低
        return False
    if abs(kp[0]) < 1e-6 and abs(kp[1]) < 1e-6:  # 坐标为零值
        return False
    
    return True

def get_kp_xy(keypoints, name):
    """安全获取关键点坐标，无效点返回None"""
    if not is_kp_valid(keypoints, name):
        return None
    idx = KEYPOINT_INDICES[name]
    return keypoints[idx][:2]