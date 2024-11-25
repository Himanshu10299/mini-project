import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # Load YOLOv5s model (smallest)

# Start webcam
cap = cv2.VideoCapture(0)  # Change to 1 if using an external webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLOv5 inference
    results = model(frame)

    # Render results
    frame = results.render()[0]  # Render boxes and labels on the frame
    cv2.imshow("Webcam Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC'
        break

cap.release()
cv2.destroyAllWindows()
