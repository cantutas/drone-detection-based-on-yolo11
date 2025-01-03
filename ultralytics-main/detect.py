# from ultralytics import YOLO

# model=YOLO('D:\\lab\\ultralytics-main\\res\\yolo-final\\weights\\best.pt',task='detect')
# res=model(source='video/test7.mp4',show=True)

import cv2
from ultralytics import YOLO
 
 
def yolo_pre():
    yolo=YOLO('D:\\lab\\ultralytics-main\\res\\runs\\new\\train25\\weights\\best.pt',task='detect')# TODO chage to your model path
    video_path='video/test7.mp4' # TODO chage to your video path
    cap = cv2.VideoCapture(video_path)  
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video/final-new.mp4', fourcc, 20.0, (frame_width, frame_height)) # TODO chage to your path
 
    while cap.isOpened():
        status, frame = cap.read()  
        if not status:
            break
        result = yolo.predict(source=frame, save=True)
        result = result[0]
        anno_frame = result.plot()
        #cv2.imshow('行人', anno_frame)
        out.write(anno_frame) 
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('保存完成')
    video_yolo_path='video/final-new.mp4'
    return video_yolo_path

yolo_pre()