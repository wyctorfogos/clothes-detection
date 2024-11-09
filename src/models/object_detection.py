import os
import cv2
from ultralytics import YOLO
import torch

class ObjectDetection():
    def __init__(self):
        self.model_path = "/weights/detect/train6/weights/last.pt"
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=(self.load_model()).to(self.device)
    
    def load_model(self):
        try:
            # Carregar o modelo de detecção de roupas
            loaded_model = YOLO(self.model_path)
            print("Modelo carregado com sucesso.")
            return loaded_model
        except Exception as e:
            raise FileExistsError("Não foi possível carregar o modelo.")
        
        

    def inference(self, image):
        labels = []
        confidences = []
        class_ids = []
        coordinates = []

        try:
            # Realizar inferência na imagem
            with torch.no_grad():
                results = self.model.predict(image)

            # Processar as detecções
            for result in results:
                for box in result.boxes:
                    # Obter coordenadas da caixa delimitadora
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    coordinates.append([x1, y1, x2, y2])

                    # Coletar confiança e ID de classe
                    confidences.append(float(box.conf[0]))
                    class_id = int(box.cls[0])
                    class_ids.append(class_id)
                    labels.append(self.model.names[class_id])

            return {
                "labels": labels,
                "confidences": confidences,
                "coordinates": coordinates,
                "class_ids": class_ids
            }
        except Exception as e:
            print(f"Erro ao realizar a inferência. Erro: {e}")
            return {
                "labels": labels,
                "confidences": confidences,
                "coordinates": coordinates,
                "class_ids": class_ids
            }
