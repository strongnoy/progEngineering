import cv2
from ultralytics import YOLO


model = YOLO('yolov8n.pt')


cap = cv2.VideoCapture("input_video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break


    results = model(frame)


    detections = results[0].boxes

    if len(detections) > 0:

        frame_with_boxes = results[0].plot()


        cv2.imshow("Detected Objects", frame_with_boxes)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
