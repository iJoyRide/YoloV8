from ultralytics import YOLO

# Load a YOLOv8 model (ensure the configuration and weights are appropriate for YOLOv8)
#model = YOLO("yolov8s-p2.yaml").load('yolov8s.pt')
#model = YOLO("/app/weights/best.pt")
model = YOLO("yolov8s.pt")
#model = YOLO("/app/weights/runs/detect/train2/weights/last.pt")
# Start training YOLOv8 with the specified hyperparameters
result = model.train(
    data="./cynapse_v2.yaml",
    batch=4,
    epochs=200,
    optimizer='SGD',
    resume = True,
    #cfg = "/app/weights/runs/detect/tune3/best_hyperparameters.yaml"
    device=0,
)
