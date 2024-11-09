import numpy as np
import cv2

def convertBufferIntoImage(image_data: bytes) -> np.ndarray:
    """
    Converte o buffer da imagem para o formato OpenCV (np.ndarray).
    
    Parâmetros:
    - image_data (bytes): Conteúdo da imagem em bytes.

    Retorno:
    - np.ndarray: Imagem no formato OpenCV ou None em caso de erro.
    """
    try:
        # Converter o buffer de bytes para um array NumPy
        np_array = np.frombuffer(image_data, np.uint8)
        # Decodificar o array em uma imagem no formato OpenCV
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Verificar se a imagem foi decodificada corretamente
        if img is None:
            raise ValueError("Falha ao decodificar a imagem.")
        
        return img
    
    except Exception as e:
        print(f"Erro ao converter o buffer para imagem: {e}")
        return None
