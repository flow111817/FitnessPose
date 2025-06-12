import time
from src.Pose_Utils import calculate_angle, get_kp_xy

class PushupCounter() :
    def __init__(self):
        self.reset()

    # 初始化
    def reset(self) :
        self.count = 0
        self.standard_count = 0
        self.status = 'up'
        self.standard_status = 'up'
        self.analysis_data = {
            'pushup_count_list': [],
            'left_angle_list': [],
            'right_angle_list': [],
            'waist_angle_list' : [],
            'time_list': [],
            'time_any' : [],
            'rhythm_speed' : None,
            'pushup_std' : None,
            'standard_ratio' : None,
            'diff_std' : None,
            'pushup_count' : None,
            'waist_down' : None
        }
        self.start_time = time.time()

    def record_action(self, keypoints):
        """记录当前动作并计算角度，处理无效点"""
        # 获取关键点坐标，处理可能的None值
        ls = get_kp_xy(keypoints, 'LEFT_SHOULDER') or [0, 0]
        le = get_kp_xy(keypoints, 'LEFT_ELBOW') or [0, 0]
        lw = get_kp_xy(keypoints, 'LEFT_WRIST') or [0, 0]
        rs = get_kp_xy(keypoints, 'RIGHT_SHOULDER') or [0, 0]
        re = get_kp_xy(keypoints, 'RIGHT_ELBOW') or [0, 0]
        rw = get_kp_xy(keypoints, 'RIGHT_WRIST') or [0, 0]
        lh = get_kp_xy(keypoints, 'LEFT_HIP') or [0, 0]
        rh = get_kp_xy(keypoints, 'RIGHT_HIP') or [0, 0]
        lk = get_kp_xy(keypoints, 'LEFT_KNEE') or [0, 0]
        rk = get_kp_xy(keypoints, 'RIGHT_KNEE') or [0, 0]

        # 计算角度，函数内部会处理无效点
        left_angle = calculate_angle(ls, le, lw)
        right_angle = calculate_angle(rs, re, rw)
        
        # 检查角度是否有效
        if left_angle < 30 or left_angle > 180:
            left_angle = 180  # 使用默认值
        if right_angle < 30 or right_angle > 180:
            right_angle = 180  # 使用默认值
        
        # 腰部角度计算
        left_waist_angle = calculate_angle(ls, lh, lk)
        right_waist_angle = calculate_angle(rs, rh, rk)
        
        # 处理无效腰部角度
        if left_waist_angle < 30 or left_waist_angle > 180:
            left_waist_angle = 180
        if right_waist_angle < 30 or right_waist_angle > 180:
            right_waist_angle = 180
        
        waist_angle_ave = (left_waist_angle + right_waist_angle) / 2 
        self.time_now = time.time() - self.start_time

        # 保存分析数据
        self.analysis_data['left_angle_list'].append(left_angle)
        self.analysis_data['right_angle_list'].append(right_angle)
        self.analysis_data['time_list'].append(self.time_now)
        self.analysis_data['waist_angle_list'].append(waist_angle_ave)

        return left_angle, right_angle
    
    def update_count(self, left_angle, right_angle):
        """根据角度更新俯卧撑计数"""
        if self.status == 'up' and left_angle < 125 and right_angle < 125:
            self.status = 'down'
            if left_angle < 90 and right_angle < 90:
                self.standard_status = 'down'
        elif self.status == 'down' and left_angle > 125 and right_angle > 125:
            self.status = 'up'
            self.count += 1
            self.analysis_data['time_any'].append(self.time_now)
            if left_angle > 160 and right_angle > 160:
                self.standard_status = 'up'
                self.standard_count += 1

        # 保存当前计数
        self.analysis_data['pushup_count_list'].append(self.count)

        return self.count, self.status

    def caculate_target(self) :
        # 动作节奏快慢得分
        speed = self.count / self.analysis_data['time_list'][-1] if self.analysis_data['time_list'] else 0
        self.analysis_data['rhythm_speed'] = 1 - 1 / (1 + 3 * speed)


        # 动作节奏稳定程度得分
        if len(self.analysis_data['time_any']) > 1:
            intervals = [self.analysis_data['time_any'][i] - self.analysis_data['time_any'][i-1] 
                        for i in range(1, len(self.analysis_data['time_any']))]
            mean_interval = sum(intervals) / len(intervals)
            std_interval = (sum((x - mean_interval)**2 for x in intervals) / len(intervals)) ** 0.5
            # 变异系数 = 标准差/平均值
            cv = std_interval / mean_interval if mean_interval > 0 else 0
            # 使用指数函数映射到0-1 (CV<0.2得高分)
            self.analysis_data['pushup_std'] = max(0, 1 - 2 * cv)
        else:
            self.analysis_data['pushup_std'] = 0.5 if self.count > 0 else 0
        

        # 动作标准率得分
        if self.count != 0 :
            self.analysis_data['standard_ratio'] = self.standard_count / self.count 
        elif self.count == 0 :
            self.analysis_data['standard_ratio'] = 0


        # 动作对称性得分
        diff = [abs(i-j) for i,j in zip(self.analysis_data['left_angle_list'], self.analysis_data['right_angle_list'])]
        mean_diff = sum(diff) / len(diff) if diff else 0
        # 使用sigmoid函数映射 (平均偏差<10°得高分)
        self.analysis_data['diff_std'] = 1 / (1 + 0.2 * mean_diff)


        # 动作个数得分
        if self.count >= 12:
                self.analysis_data['pushup_count'] = 1
        elif self.count > 0:
            # 渐进式: 1个得0.1分，6个得0.5分，11个得0.9分
            self.analysis_data['pushup_count'] = 1 - 1 / (1 + 0.2 * self.count)
        else:
            self.analysis_data['pushup_count'] = 0


        # 塌腰得分
        waist_dev = [abs(180 - x) for x in self.analysis_data['waist_angle_list']]
        mean_waist_dev = sum(waist_dev) / len(waist_dev) if waist_dev else 0
        # 使用sigmoid函数映射 (平均偏差<10°得高分)
        self.analysis_data['waist_down'] = 1 / (1 + 0.1 * mean_waist_dev)

        return self.analysis_data
    