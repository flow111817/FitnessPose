import cv2
from config import KEYPOINT_INDICES

def draw_skeleton(frame, keypoints):
    """在帧上绘制骨架"""
    # 关键点连接关系
    CONNECTIONS = [
        ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
        ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
        ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
        ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP'),
        ('LEFT_HIP', 'RIGHT_HIP'), ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
        ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
        ('NOSE', 'LEFT_EYE'), ('NOSE', 'RIGHT_EYE'),
        ('LEFT_EYE', 'LEFT_EAR'), ('RIGHT_EYE', 'RIGHT_EAR')
    ]

    # 绘制关键点
    for x, y, v in keypoints:
        if v > 0.5:  # 置信度阈值
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # 绘制连接线
    for (start, end) in CONNECTIONS:
        start_idx = KEYPOINT_INDICES[start]
        end_idx = KEYPOINT_INDICES[end]
        
        if keypoints[start_idx][2] > 0.5 and keypoints[end_idx][2] > 0.5:
            start_pt = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_pt = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            cv2.line(frame, start_pt, end_pt, (255, 255, 0), 2)
    
    return frame

def display_info(frame, count, left_angle, right_angle, status):
    """在帧上显示计数和角度信息"""
    cv2.putText(frame, f"Pushups: {count}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Left Angle: {left_angle:.1f}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Right Angle: {right_angle:.1f}", (20, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    return frame