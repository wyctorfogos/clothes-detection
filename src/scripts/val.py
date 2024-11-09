import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


def getFinalImage(image):
    # Carregar o modelo de detecção de roupas
    loaded_model = YOLO(model_path)
    print("Modelo carregado com sucesso.")
    
    # Realizar inferência na imagem de teste
    results = loaded_model.predict(image_path)
    # Desenhar as detecções na imagem
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Obter as coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = int(box.cls[0])
            label = loaded_model.names[class_id]
            
            # Desenhar a caixa
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Adicionar o rótulo e a confiança
            cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Converter a imagem de BGR para RGB para exibição correta com matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Exibir a imagem com as detecções
    plt.figure(figsize=(10, 8))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Detecções de Roupas")
    plt.show()


if __name__=="__main__":
    # Caminho para o modelo treinado
    model_path = "/home/wytcor/PROJECTs/ClothesDetectionAndRecognition/runs/detect/train6/weights/last.pt"
    # Caminho para a imagem de teste
    image_path = "/home/wytcor/PROJECTs/ClothesDetectionAndRecognition/src/image_to_test.jpg"  # Atualize este caminho conforme necessário
    # Verificar se o modelo existe
    if not os.path.isfile(model_path):
        print(f"Erro: O modelo '{model_path}' não foi encontrado.")
        exit(1)

    # Verificar se a imagem existe
    if not os.path.isfile(image_path):
        print(f"Erro: A imagem '{image_path}' não foi encontrada.")
        exit(1)

    # Carregar a imagem
    image = cv2.imread(image_path)

    if image is None:
        print(f"Erro: Não foi possível carregar a imagem '{image_path}'. Verifique o caminho e o formato da imagem.")
        exit(1)
    else:
        print("Imagem carregada com sucesso.")

    
    # Load model with inferences
    getFinalImage(image)