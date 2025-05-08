import torch
import sys
sys.path.append('./yolov5')  # Adiciona o diretório yolov5 ao path

# Carrega o modelo YOLOv5 local
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
import cv2
import numpy as np

def test_camera():
    # Inicializa o dispositivo
    device = select_device('')  # '' para CPU, '0' para primeira GPU

    # Carrega o modelo
    model = DetectMultiBackend('yolov5s.pt', device=device)
    stride = model.stride
    names = model.names

    # Configura a webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Captura frame da webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Prepara a imagem
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1).float()  # HWC to CHW
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inferência
        pred = model(img)
        pred = non_max_suppression(pred)[0]

        # Processa detecções
        if len(pred):
            # Redimensiona as coordenadas para o tamanho original
            pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], frame.shape).round()

            # Desenha as detecções
            for *xyxy, conf, cls in pred:
                label = f'{names[int(cls)]} {conf:.2f}'
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostra o resultado
        cv2.imshow('YOLOv5 Detection', frame)

        # Sai com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()