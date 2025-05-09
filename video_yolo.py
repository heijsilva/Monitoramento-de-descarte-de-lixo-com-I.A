import cv2
import torch

# Caminho para o modelo treinado (ajustado para rodar da raiz do projeto)
MODEL_PATH = 'yolov5/runs/train/exp7/weights/best.pt'
VIDEO_PATH = 0  # 0 = webcam padrão

# Carrega o modelo YOLOv5 customizado
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# Abre a webcam
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Faz a predição
    results = model(frame)
    annotated_frame = results.render()[0]

    # Mostra o frame com as detecções
    cv2.imshow('YOLOv5 - Webcam Detection', annotated_frame)

    # Sai ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()