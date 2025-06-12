from config import MAX_ABNORMAL_FRAMES,ALPHA,MAX_DELTA
import numpy as np

class FrameOptimizer() :
    def __init__(self):
        self.abnormal_counts  = [0] * 17
        self.prev_kps = None

    # 异常关键点过滤
    def filter_abnormal_keypoints(self,current_kps, max_delta=MAX_DELTA):

        if self.prev_kps is None:
            return current_kps

        filtered = []

        for i, (curr, prev) in enumerate(zip(current_kps, self.prev_kps)):
            dx = abs(curr[0] - prev[0])
            dy = abs(curr[1] - prev[1])
            dist = (dx**2 + dy**2)**0.5

            if dist > max_delta:
                self.abnormal_counts[i] += 1
                if self.abnormal_counts[i] >= MAX_ABNORMAL_FRAMES:
                    # 连续异常太久，接受当前预测
                    filtered.append(curr)
                    self.abnormal_counts[i] = 0  # 重置
                else:
                    # 仍然使用上一帧
                    filtered.append([prev[0], prev[1], curr[2]])
            else:
                self.abnormal_counts[i] = 0  # 正常时重置
                filtered.append(curr)

        return filtered


    # 指数加权滑动平均
    def smooth_keypoints(self, current_kps, alpha=ALPHA):

        if self.prev_kps is None:
            return current_kps
        
        current_kps = np.array(current_kps)
        self.prev_kps = np.array(self.prev_kps)
        smoothed = alpha * self.prev_kps + (1 - alpha) * current_kps
        
        return smoothed.tolist()
