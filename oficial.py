from ultralytics import YOLO
import cv2

model = YOLO('yolov8m.pt')

video = cv2.VideoCapture('IMG_1240.mp4')

line1 = [(181, 252), (379, 94)]
line2 = [(265, 328), (472, 123)]
line3 = [(223,291) , (423,111)]

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def pessoa_na_faixa(pessoa_box, line):
    diag_line = (pessoa_box[:2].astype(int), pessoa_box[2:].astype(int))
    intercept = intersect(diag_line[0], diag_line[1], line[0], line[1])
    return intercept

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    resultados = model(frame, classes=[0], verbose=False)

    contador_pessoas = 0

    contador_pessoas = 0
    contador_anterior = 0

    for pessoa_box in resultados:
        for box in pessoa_box:
            b = box.boxes.xyxy[0].cpu().numpy()
            if pessoa_na_faixa(b, line1) or pessoa_na_faixa(b, line2) or pessoa_na_faixa(b, line3):
                cv2.rectangle(frame, b[:2].astype(int), b[2:].astype(int), (0, 0, 255), 2)
                cv2.putText(frame, "Entrou!!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                contador_pessoas += 1
        cv2.putText(frame, "Contagem: " + str(contador_pessoas), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()