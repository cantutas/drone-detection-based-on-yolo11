from ultralytics import YOLO

# model=YOLO('D:\\lab\\ultralytics-main\\ultralytics\\cfg\\models\\11\\yolo11n-CBAM.yaml').load('D:\\lab\\ultralytics-main\\yolo11s.pt')
# #model=YOLO('./yolo11s.pt')

# model.train(data='yolo-bvn.yaml',workers=0,epochs=200,batch=64)

model=YOLO('D:\\lab\\ultralytics-main\\res\\runs\\new\\train21\\weights\\best.pt')
metrics=model.val(split='test',data='yolo-bvn.yaml')