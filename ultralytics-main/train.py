from ultralytics import YOLO

model=YOLO('ultralytics\\cfg\models\\11\\yolo11-final.yaml')

model.train(data='yolo-bvn.yaml',workers=0,epochs=200,batch=64)

# model=YOLO('weights\\best.pt')
# metrics=model.val(split='test',data='yolo-bvn.yaml')
