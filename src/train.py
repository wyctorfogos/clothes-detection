import os
import cv2
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt


# importa os modelos e usa os dados de 'data'
model = YOLO("yolov8m.pt")
results=model.train(data="/home/wytcor/PROJECTs/ClothesDetectionAndRecognition/data/Clothes detection.v4i.yolov8/data.yaml", epochs=100, batch=32, imgsz=64, scale=0.1, mosaic=0.25)
