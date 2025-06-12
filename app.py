import os
import uuid
import cv2
from ultralytics import YOLO
from config import MODEL_PATH
from src.Report_generator import Generater
from src.Pushup_Counter import PushupCounter
from src.Visualization import draw_skeleton,display_info
from src.Frame_Optimizer import FrameOptimizer
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB 限制
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_video(video_path):
    """处理视频并生成报告"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "无法打开视频文件"
    
    # 获取视频信息
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化组件
    model = YOLO(MODEL_PATH)
    pushup_counter = PushupCounter()
    frame_optimizer = FrameOptimizer()
    generater = Generater(frame_width, frame_height)
    
    # 处理每一帧
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整大小（可选）
        frame = cv2.resize(frame, (640, 480))
        
        # 姿态检测
        results = model.predict(frame, verbose=False)
        
        try:
            # 获取关键点
            keypoints = results[0].keypoints.data[0].cpu().numpy().tolist()
            keypoints = frame_optimizer.filter_abnormal_keypoints(keypoints)
            keypoints = frame_optimizer.smooth_keypoints(keypoints)
            frame_optimizer.prev_kps = keypoints
            
            # 分析动作
            l_angle, r_angle = pushup_counter.record_action(keypoints)
            pushup_count, status = pushup_counter.update_count(l_angle, r_angle)
            
            # 可视化
            frame = draw_skeleton(frame, keypoints)
            frame = display_info(frame, pushup_count, l_angle, r_angle, status)
            
            generater.add_frame(frame)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"已处理 {frame_count} 帧...")

        except Exception as e:
            print(f"处理错误: {e}")
            continue

    # 释放资源
    cap.release()
    
    # 生成报告和视频
    analysis_data = pushup_counter.caculate_target()
    report_path = generater.generate_report(analysis_data)
    video_path = generater.generate_video()
    
    return analysis_data, video_path,report_path

@app.route('/')
def index():
    """首页 - 上传视频"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """处理视频上传"""
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # 生成唯一文件名
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 处理视频
        analysis_data, video_path, report_path = process_video(filepath)
        
        if analysis_data is None:
            return render_template('index.html', error="视频处理失败")
        
        # 保存结果信息
        result_id = uuid.uuid4().hex
        app.config['results'][result_id] = {
            'analysis': analysis_data,
            'video': video_path,
            'report' : report_path
        }
        
        return redirect(url_for('show_results', result_id= result_id))
    
    return render_template('index.html', error="不支持的文件类型")


@app.route('/result/<result_id>')
def show_results(result_id):
    """显示分析结果"""
    if result_id not in app.config['results']:
        return redirect(url_for('index'))
    
    result = app.config['results'][result_id]
    return render_template('results.html', report_path = result['report'][7:],
                           analysis=result['analysis'],
                           video_path=result['video'][7:])

@app.route('/data/<path:filename>')
def serve_data(filename):
    return send_from_directory('data', filename)

@app.route('/video/<result_id>')
def show_video(result_id):
    """显示处理后的视频"""
    if result_id not in app.config['results']:
        return redirect(url_for('index'))
    
    result = app.config['results'][result_id]
    return render_template('video.html', 
                          video_url=url_for('static', filename=result['video']))

if __name__ == '__main__':
    # 初始化结果存储
    app.config['results'] = {}
    app.run(host='0.0.0.0', port=5000 ,debug=True)

