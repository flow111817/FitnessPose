import matplotlib.pyplot as plt
from datetime import datetime 
from scipy.ndimage import gaussian_filter1d
from config import REPORT_PATH,VIDEO_PATH
import cv2
import numpy as np
import matplotlib.font_manager as fm

class Generater() :
    def __init__(self,frame_width,frame_height):
        self.frames = []
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.report_name = REPORT_PATH + f"report_{datetime.now().strftime('%H%M%S')}.png"
        self.video_name = VIDEO_PATH + f"video_{datetime.now().strftime('%H%M%S')}.mp4"
        
    def generate_report(self,analysis_data):
        """生成并保存分析报告图表"""

        smoothed_left = gaussian_filter1d(analysis_data['left_angle_list'], sigma=2)
        smoothed_right = gaussian_filter1d(analysis_data['right_angle_list'], sigma=2)
        
        fig = plt.figure(figsize=(8,8))
        
        # 角度变化图
        plt.subplot(2,2,(1,2))
        plt.plot(analysis_data['time_list'], smoothed_left, 
                color='C1', label='left angle')
        plt.plot(analysis_data['time_list'], smoothed_right, 
                color='C0', label='right angle')
        plt.axhline(y=90, color='r', linestyle='--', label='low threshold')
        plt.axhline(y=160, color='g', linestyle='--', label='upper threshold')
        plt.xlabel('time (s)')
        plt.ylabel('angle (°)')
        plt.legend()
        plt.grid(True)
        
        # 计数变化图
        plt.subplot(2,2,3)
        plt.plot(analysis_data['time_list'], analysis_data['pushup_count_list'], 
                color='C1')
        plt.xlabel('time (s)')
        plt.ylabel('pushup counts')
        plt.grid(True)
        
        # 雷达图
        ax4 = fig.add_subplot(2, 2, 4, polar=True)
        categories = [
            'rhythm',
            'rhythm stability', 
            'standard', 
            'balanced', 
            'number', 
            'Trunk stability'
            ]
        values = [analysis_data['rhythm_speed'],
                  analysis_data['pushup_std'],
                  analysis_data['standard_ratio'], 
                  analysis_data['diff_std'],
                  analysis_data['pushup_count'], 
                  analysis_data['waist_down']]

        # 将类别转换为角度
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        values += values[:1]
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)

        # 绘制雷达图
        ax4.plot(angles, values, linewidth=2, linestyle='solid')
        ax4.fill(angles, values, alpha=0.25)

        # 添加标题和图例
        ax4.set_title('Capacity')

        plt.tight_layout()
        plt.savefig(self.report_name, dpi=300)
        plt.close()
        
        print(f'报告保存成功: {self.report_name}')

        return self.report_name
    
    def add_frame(self, frame):
        self.frames.append(frame.copy())

    def generate_video(self) :
        if not self.frames:
            print("没有帧可保存")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_name, fourcc, 25 , (640,480))
        for frame in self.frames:
            out.write(frame)
        out.release()
        print(f"视频保存成功: {self.video_name}")

        return self.video_name