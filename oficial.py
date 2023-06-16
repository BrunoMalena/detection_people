from ultralytics import YOLO
import cv2

model = YOLO('yolov8m.pt')

video = cv2.VideoCapture('IMG_1214.mp4')

frame6 = (130, 310)  
frame7 = (460, 300)  
frame8 = (400, 260)  
frame9 = (100, 230)  

def pessoa_na_faixa(pessoa_box, faixa_box):
    
    x1, y1, x2, y2 = pessoa_box
    fx1, fy1, fx2, fy2 = faixa_box
    if x2 < fx1 or fx2 < x1 or y2 < fy1 or fy2 < y1:
        return False
    else:
        return True

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

    faixa_pedestres = (x_min, y_min, x_max, y_max) 

    for pessoa_box in resultados:
        for box in pessoa_box:
            b = box.boxes.xyxy[0].tolist()
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            if pessoa_na_faixa((x1, y1, x2, y2), faixa_pedestres):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()