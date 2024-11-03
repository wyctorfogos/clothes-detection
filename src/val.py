import os
import cv2
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Carregar o modelo da detecção de roupas
loaded_model= YOLO("/home/wytcor/PROJECTs/ClothesDetectionAndRecognition/runs/detect/train6/weights/best.pt")

# Carregar a imagem de teste
results=loaded_model.pred("/home/wytcor/PROJECTs/ClothesDetectionAndRecognition/data/Clothes detection.v4i.yolov8/test/images/img_0473_jpeg.rf.c4ced0be4c6a24695cbef27c4fe24f8d.jpg")