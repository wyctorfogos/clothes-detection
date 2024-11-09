import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import utils
from utils.convertBufferIntoImage import convertBufferIntoImage
import models
from models import object_detection
import os

API_PORT=os.getenv("API_PORT")

# Instanciar o modelo de detecção de objetos
model = object_detection.ObjectDetection()

# Criar a aplicação FastAPI
app = FastAPI()

# Rota para detecção de roupas
@app.post('/clothes_detection')
async def clothes_detection(image: UploadFile = File(...)):
    try:
        # Ler a imagem enviada como arquivo
        image_data = await image.read()

        if not image_data:
            raise HTTPException(status_code=400, detail="Nenhuma imagem enviada.")
        
        # Converter o conteúdo da imagem para um formato processável (array NumPy para OpenCV)
        img = convertBufferIntoImage(image_data=image_data)
        if img is None:
            raise HTTPException(status_code=422, detail="Erro ao converter o conteúdo da imagem.")
        
        # Realizar a inferência usando o modelo
        result = model.inference(img)

        # Retornar o resultado da inferência
        return {"status": "success", "data": result}
    
    except Exception as e:
        print(f"Erro ao processar a requisição: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar a requisição")

# Iniciar o servidor Uvicorn se o script for executado diretamente
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(API_PORT))
