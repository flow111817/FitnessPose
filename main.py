from ultralytics import YOLO
import cv2
from config import MODEL_PATH,CAMERA_INDEX
from src.Report_generator import Generater
from src.Pushup_Counter import PushupCounter
from src.Visualization import draw_skeleton,display_info
from src.Frame_Optimizer import FrameOptimizer

# 主程序
def main() : 

    # 初始化
    cap = cv2.VideoCapture(CAMERA_INDEX)
    model = YOLO(MODEL_PATH)
    pushup_counter = PushupCounter()
    frame_optimizer = FrameOptimizer()

    # 获取帧宽高用于后续视频写入
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    generater = Generater(frame_width,frame_height)

    # 主循环
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("帧获取失败")
            break

        frame = cv2.resize(frame, (640, 480))
        results = model.predict(frame, verbose=False)

        try :
            keypoints = results[0].keypoints.data[0].cpu().numpy().tolist()
            if len(keypoints) < 17:
                keypoints = [[0, 0, 0] for _ in range(17)]
            else:
                # 填充缺失的点
                keypoints = [k if len(k) == 3 else [0, 0, 0] for k in keypoints]

            # 获取关键点
            
            keypoints = frame_optimizer.filter_abnormal_keypoints(keypoints)
            keypoints = frame_optimizer.smooth_keypoints(keypoints)
            frame_optimizer.prev_kps = keypoints

            # 分析并可视化
            l_angle, r_angle = pushup_counter.record_action(keypoints)
            pushup_count, status = pushup_counter.update_count(l_angle, r_angle)
            frame = draw_skeleton(frame, keypoints)
            frame = display_info(frame, pushup_count, l_angle, r_angle, status)
            
            generater.add_frame(frame)

            cv2.imshow("YOLOv8 Pose Pushup", frame)

            if cv2.waitKey(10) & 0xFF == ord(' '):
                break
            
        # 错误处理
        except Exception as e:
            print(f"处理错误: {e}")

            # 显示当前帧而不处理
            cv2.imshow("YOLOv8 Pose Pushup", frame)
            if cv2.waitKey(10) & 0xFF == ord(' '):
                break
            continue

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    # 生成报告和视频
    analysis_data = pushup_counter.caculate_target()
    report_path = generater.generate_report(analysis_data)
    video_path = generater.generate_video()

if __name__ == '__main__' :
    main()

