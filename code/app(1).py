import datetime
from flask import Flask, render_template, request, send_file,Response
from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/record', methods=['POST'])
def record():
    
    # 存储追踪历史
    track_history = defaultdict(lambda: [])
    cap = cv2.VideoCapture(0)
    def generate_frames2():
        
        model = YOLO('check_point/rider_yolov8.pt')
        pedestrians=0
        riders=0
        cars=0
        pedestrians_sum=0
        riders_sum=0
        cars_sum=0
        id_cls_mapping = {}
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while True:
        
            # 从视频读取一帧
            success, frame = cap.read()
               # 计算平均亮度
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            average_brightness = cv2.mean(gray_img)[0]
            print(average_brightness)

            if(average_brightness < 90):
                    # frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
                    # cv2.imwrite(frame_filename, frame)
                    # frame_count += 1
                    # frame_filename = os.path.join(output_folder, f'frame_{frame_count}.jpg')
                    B,G,R = cv2.split(frame) #get single 8-bits channel
                    b=cv2.equalizeHist(B)
                    g=cv2.equalizeHist(G)
                    r=cv2.equalizeHist(R)
                    equal_img=cv2.merge((b,g,r))  #merge it back
                    # cv2.imwrite(frame_filename, equal_img)
                    frame=equal_img

            if success:
                # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
                results = model.track(frame, persist=True,tracker='cfg/trackers/botsort.yaml')
                # 获取框和追踪ID
                boxes = results[0].boxes.xywh.cpu()
                if results[0].boxes.id!=None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.cpu().tolist()
                    for track_id, cls in zip(track_ids, clss):
                            if cls in id_cls_mapping:
                                id_cls_mapping[cls].add(track_id)
                            else:
                                id_cls_mapping[cls] = {track_id}
                    # 统计cls为0和1对应的id数量
                    pedestrians_sum = id_cls_mapping.get(0, set())
                    pedestrians_sum=len(pedestrians_sum)
                    riders_sum = id_cls_mapping.get(1, set())
                    riders_sum=len(riders_sum)
                    cars_sum = id_cls_mapping.get(3, set())
                    cars_sum=len(cars_sum)
                    # print(id_cls_mapping)

                    for value in clss:
                        if value == 0.0:
                            pedestrians += 1
                        if value == 1.0:
                            riders += 1
                        if value == 3.0:
                            cars += 1
                    # 在帧上展示结果
                    annotated_frame = results[0].plot(conf=False)
                    # 绘制追踪路径
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y中心点
                        if len(track) > 30:  # 在90帧中保留90个追踪点
                            track.pop(0)

                        # 绘制追踪线
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        position = (10, 30)  # 左上角位置
                        position2 = (10, 60)  # 左上角位置
                        position3 = (1000, 30)  # 右上角位置
                        position4 = (1000, 60)  # 右上角位置
                        position5 = (10, 90)  # 左上角位置
                        position6 = (1000, 90)  # 右上角位置
                        font_scale = 1
                        font_color = (0, 0, 255)  # 字体颜色为红色
                        font_color2 = (255, 255, 255)  # 字体颜色为白色
                        line_type = 2
                        # 在图像上叠加文本
                        cv2.putText(annotated_frame, f'pedestrians: {pedestrians}', position, font, font_scale, font_color, line_type)
                        cv2.putText(annotated_frame, f'riders: {riders}', position2, font, font_scale, font_color, line_type)
                        cv2.putText(annotated_frame, f'cars: {cars}', position5, font, font_scale, font_color, line_type)
                        cv2.putText(annotated_frame, f'pedestrians_sum: {pedestrians_sum}', position3, font, font_scale, font_color2, line_type)
                        cv2.putText(annotated_frame, f'riders_sum: {riders_sum}', position4, font, font_scale, font_color2, line_type)
                        cv2.putText(annotated_frame, f'cars_sum: {cars_sum}', position6, font, font_scale, font_color2, line_type)
                    
                    # 展示带注释的帧
                    pedestrians=0
                    riders=0
                    cars=0
                    #cv2.imshow("YOLOv8 Tracking", annotated_frame)
                    ret, buffer = cv2.imencode('.jpg', annotated_frame)
                    annotated_frame = buffer.tobytes()
                    yield (b'--frame\r\n' 
                        b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame + b'\r\n')

                    # 如果按下'q'则退出循环
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                # 如果视频结束则退出循环
                break
    return Response(generate_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')




# 导入视频
@app.route('/uploadDIP', methods=['POST'])
def uploadDIP():
    file = request.files['file']
    if file:
        filename = "original_"+file.filename

        # 构建新的文件路径，将文件保存到app.py所在文件夹
        upload_folder = os.path.dirname(__file__)
        file_path = os.path.join(upload_folder, filename)

        # 保存上传的文件到新的文件路径
        file.save(file_path)

        return render_template('index.html', filename1=filename)
    else:
        return "No file uploaded."

@app.route('/video/<filename>')
def video(filename):
    return send_file(filename, mimetype='video/mp4')






# 图像处理
@app.route('/play_DIP', methods=['POST'])
def play_DIP():
    # 从POST请求中获取文件名
    original_filename = request.form.get('filename1')
    # 构建新的文件路径，将文件保存到app.py所在文件夹
    upload_folder = os.path.dirname(__file__)
    # 定义原始音频的路径和降噪后的音频的路径
    
    #####
    output_path = os.path.join(upload_folder, original_filename)



    # 打开视频文件
    
    video_path = output_path
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "output/tracking_result.mp4"
    
    
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can try other codecs like 'XVID' based on your system support
    # out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    # 存储追踪历史
    track_history = defaultdict(lambda: [])

    # 循环遍历视频帧
    
    def generate_frames():
        
        model = YOLO('check_point/rider_yolov8.pt')
        pedestrians=0
        riders=0
        cars=0
        pedestrians_sum=0
        riders_sum=0
        cars_sum=0
        id_cls_mapping = {}
        
        while cap.isOpened():
            # 从视频读取一帧
            success, frame = cap.read()
               # 计算平均亮度
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            average_brightness = cv2.mean(gray_img)[0]
            print(average_brightness)

            if(average_brightness < 90):
                    B,G,R = cv2.split(frame) #get single 8-bits channel
                    b=cv2.equalizeHist(B)
                    g=cv2.equalizeHist(G)
                    r=cv2.equalizeHist(R)
                    equal_img=cv2.merge((b,g,r))  #merge it back
                    # cv2.imwrite(frame_filename, equal_img)
                    frame=equal_img

            if success:
                # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
                results = model.track(frame, persist=True,tracker='cfg/trackers/botsort.yaml')
                # 获取框和追踪ID
                boxes = results[0].boxes.xywh.cpu()
                if results[0].boxes.id!=None:
                    track_ids = results[0].boxes.id.int().cpu().tolist()
                    clss = results[0].boxes.cls.cpu().tolist()

                    for track_id, cls in zip(track_ids, clss):
                            if cls in id_cls_mapping:
                                id_cls_mapping[cls].add(track_id)
                            else:
                                id_cls_mapping[cls] = {track_id}
                    # 统计cls为0和1对应的id数量
                    pedestrians_sum = id_cls_mapping.get(0, set())
                    pedestrians_sum=len(pedestrians_sum)
                    riders_sum = id_cls_mapping.get(1, set())
                    riders_sum=len(riders_sum)
                    cars_sum = id_cls_mapping.get(3, set())
                    cars_sum=len(cars_sum)
                    # print(id_cls_mapping)

                    for value in clss:
                        if value == 0.0:
                            pedestrians += 1
                        if value == 1.0:
                            riders += 1
                        if value == 3.0:
                            cars += 1
                    # 在帧上展示结果
                    annotated_frame = results[0].plot(conf=False,font_size=1280)

                    # 绘制追踪路径
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y中心点
                        if len(track) > 30:  # 在90帧中保留90个追踪点
                            track.pop(0)

                        # 绘制追踪线
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        position = (10, 30)  # 左上角位置
                        position2 = (10, 55)  # 左上角位置
                        position3 = (900, 30)  # 右上角位置
                        position4 = (900, 55)  # 右上角位置
                        position5 = (10, 80)  # 左上角位置
                        position6 = (900, 80)  # 右上角位置
                        font_scale = 1
                        font_color = (0, 0, 255)  # 字体颜色为红色
                        font_color2 = (255, 255, 255)  # 字体颜色为白色
                        line_type = 2
                        # 在图像上叠加文本
                        cv2.putText(annotated_frame, f'pedestrians: {pedestrians}', position, font, font_scale, font_color, line_type)
                        cv2.putText(annotated_frame, f'riders: {riders}', position2, font, font_scale, font_color, line_type)
                        cv2.putText(annotated_frame, f'cars: {cars}', position5, font, font_scale, font_color, line_type)
                        cv2.putText(annotated_frame, f'pedestrians_sum: {pedestrians_sum}', position3, font, font_scale, font_color2, line_type)
                        cv2.putText(annotated_frame, f'riders_sum: {riders_sum}', position4, font, font_scale, font_color2, line_type)
                        cv2.putText(annotated_frame, f'cars_sum: {cars_sum}', position6, font, font_scale, font_color2, line_type)
                    # out.write(annotated_frame)
                    # 展示带注释的帧
                    pedestrians=0
                    riders=0
                    cars=0
                    #cv2.imshow("YOLOv8 Tracking", annotated_frame)
                    ret, buffer = cv2.imencode('.jpg', annotated_frame)
                    annotated_frame = buffer.tobytes()
                    yield (b'--frame\r\n' 
                        b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame + b'\r\n')
                    # 如果按下'q'则退出循环
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                # 如果视频结束则退出循环
                break
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    


if __name__ == '__main__':
    app.run()