from ultralytics import YOLO

# 載入原本的 .pt 模型
model = YOLO("yolov8s_pose_license_plate.pt")

# 重新匯出，重點在於指定 opset=17 (或 19)
# 這樣產生的 ONNX 就能在現有的 onnxruntime 中順利開啟
# imgsz=640(這樣就是 640x640，同等於 [640, 640]) 匯出大小，固定之後，在使用推論時，就必須也是一致的
# 因此原本的 .pt 檔大小是多少，這邊也設定一致的大小
model.export(format="onnx", opset=17, imgsz=640)
