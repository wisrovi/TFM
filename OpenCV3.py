import numpy as np
import time
import cv2
import os.path as path

class OpenCV_3():
    """
    Esta es una manera que no utiliza el tradicional Haar de OpenCV, pues al ser basados en FaceRest es algo lento
    y aumentar la velocidad del reconocimiento es posible con el LBP de OpenCV, se pierde presición
    para ello, se presenta un modelo de DeepLearning que usa los algoritmos de FaceNet de google
    para buscar los rostros de manera optima y rapida
    """    
    def __init__(self):
        protocolo = "deploy.prototxt.txt"
        model = "res10_300x300_ssd_iter_140000.caffemodel"
        if not path.exists(model):
            print("No se encuetra el archivo model")
        elif not path.exists(protocolo):
            print("No se encuetra el archivo protocolo")
        else:
            self.net = cv2.dnn.readNetFromCaffe(protocolo, model)
        self.confianza = 0.55
        
    def setConfianzaDeteccion(self, confianza):
        self.confianza = confianza
    
    def detectarRostros(self, imagen):
        (h, w) = imagen.shape[:2]
        blob = cv2.dnn.blobFromImage(
                cv2.resize(imagen, (300, 300)),
                1.0,
                (300, 300),
                (104.0, 177.0, 123.0)
            )     
        self.net.setInput(blob)
        self.detections = self.net.forward()
        listadoCoordenadasRostro = []
        for i in range(0, self.detections.shape[2]):
            confianzaEsteRostro = self.detections[0, 0, i, 2]
            if confianzaEsteRostro >= self.confianza:
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                coordenadas = (startX, startY, endX, endY, confianzaEsteRostro)
                listadoCoordenadasRostro.append(coordenadas)
                
        return listadoCoordenadasRostro