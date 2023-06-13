from ultralytics import YOLO
import cv2

model = YOLO('yolov8m.pt')

video = cv2.VideoCapture('IMG_1214.mp4')

frame6 = (120, 300)  
frame7 = (520, 300)  
frame8 = (420, 230)  
frame9 = (100, 230)   

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    resultados = model(frame, classes=[0], verbose=False)

    # frame = cv2.circle(frame, frame6, 5, (255, 0, 0), -1)
    # frame = cv2.circle(frame, frame7, 5, (255, 0, 0), -1)
    # frame = cv2.circle(frame, frame8, 5, (255, 0, 0), -1)
    # frame = cv2.circle(frame, frame9, 5, (255, 0, 0), -1)

    # frame1 = cv2.line(frame, frame6, frame9, (0, 255, 0), 2)
    # frame2 = cv2.line(frame, frame8, frame9, (0, 255, 0), 2)
    # frame3 = cv2.line(frame, frame8, frame7, (0, 255, 0), 2)
    # frame4 = cv2.line(frame, frame6, frame7, (0, 255, 0), 2)

    x_min = min(frame6[0], frame7[0], frame8[0], frame9[0])
    x_max = max(frame6[0], frame7[0], frame8[0], frame9[0])
    y_min = min(frame6[1], frame7[1], frame8[1], frame9[1])
    y_max = max(frame6[1], frame7[1], frame8[1], frame9[1])

    for r in resultados:
        for box in r:
            b = box.boxes.xyxy[0].tolist()
            objeto_x = (b[0] + b[2]) / 2
            objeto_y = (b[1] + b[3]) / 2

            if x_min < objeto_x < x_max and y_min < objeto_y < y_max:
                frame = cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()