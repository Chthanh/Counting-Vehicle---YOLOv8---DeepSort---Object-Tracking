from ultralytics import YOLO
from tracker import Tracker
import cv2
import cvzone
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("test.mp4")
cap.set(3, 1280)
cap.set(4, 720)

limits = [445,500,730,520]

counter = []

tracker = Tracker()
mask = cv2.imread('mask.png')

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream = True)
    for result in results:
        detections = []
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2, conf, cls = int(x1), int(y1), int(x2), int(y2), round(float(conf), 2), int(cls)

            if (cls == 2 or cls == 7) and conf > 0.3:
                cvzone.putTextRect(img, f'{conf}', (max(0, x1), max(35, y1)), scale = 0.7, thickness = 1,offset = 3)
                detections.append([x1, y1, x2, y2, conf])

        tracker.update(img, detections)


        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cvzone.cornerRect(img, bbox= (x1, y1, x2-x1, y2-y1), t = 2)
            #cvzone.putTextRect(img, f'{track_id}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=3)

            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w//2, y1 + h//2
            if limits[0] < cx < limits[2] and limits[1] < cy < limits[3]:
                if counter.count(track_id) == 0:
                    counter.append(track_id)

    cvzone.putTextRect(img, f'Count: {len(counter)}', (50,50))
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,0,255),5)
    cv2.imshow("Image", img)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()







