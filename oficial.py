from ultralytics import YOLO
import cv2

model = YOLO('yolov8m.pt')

line1 = [(181, 252), (379, 94)]  # Define a primeira linha da faixa de pedestres
line2 = [(265, 328), (472, 123)]  # Define a segunda linha da faixa de pedestres
line3 = [(223, 291), (423, 111)]  # Define a terceira linha da faixa de pedestres

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])  # Verifica se três pontos estão dispostos no sentido anti-horário

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)  # Verifica se dois segmentos de linha se interceptam

def pessoa_na_faixa(pessoa_box, line):
    diag_line = (pessoa_box[:2].astype(int), pessoa_box[2:].astype(int))  # Converte as coordenadas do retângulo em duas coordenadas diagonais
    intercept = intersect(diag_line[0], diag_line[1], line[0], line[1])  # Verifica se o retângulo intercepta a linha da faixa de pedestres
    return intercept

video = cv2.VideoCapture('IMG_1240.mp4')  # Abre o arquivo de vídeo

while video.isOpened():
    ret, frame = video.read()  # Lê o próximo frame do vídeo
    if not ret:
        break  # Sai do loop se não houver mais frames para ler

    resultados = model(frame, classes=[0], verbose=False)  # Executa a detecção de objetos no frame

    contador_pessoas = 0  # Inicializa o contador de pessoas na faixa de pedestres

    for pessoa_box in resultados:
        for box in pessoa_box:
            b = box.boxes.xyxy[0].cpu().numpy()  # Obtém as coordenadas do retângulo delimitador da pessoa detectada
            if pessoa_na_faixa(b, line1) or pessoa_na_faixa(b, line2) or pessoa_na_faixa(b, line3):  # Verifica se a pessoa está na faixa de pedestres
                cv2.rectangle(frame, b[:2].astype(int), b[2:].astype(int), (0, 0, 255), 2)  # Desenha um retângulo em volta da pessoa
                contador_pessoas += 1  # Incrementa o contador de pessoas

        cv2.putText(frame, "Contagem: " + str(contador_pessoas), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Exibe a contagem de pessoas na faixa

    cv2.imshow('Video', frame)  # Exibe o frame com as marcações

    if cv2.waitKey(1) == ord('q'):
        break  # Sai do loop se a tecla 'q' for pressionada

video.release()
cv2.destroyAllWindows()