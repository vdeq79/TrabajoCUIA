import cv2
import numpy as np
import sys
import os

if os.path.exists('camara.py'):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()

from model import *

lena = cv2.imread("../lena.tif")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Tamaño del frame de la cámara: ", wframe, "x", hframe)

    matrix, roi = cv2.getOptimalNewCameraMatrix(camara.cameraMatrix, camara.distCoeffs, (wframe,hframe), 1, (wframe,hframe))
    roi_x, roi_y, roi_w, roi_h = roi

    final = False
    while not final:
        ret, framebgr = cap.read()

        if ret:
            # Aquí procesamos el frame
            framerectificado = cv2.undistort(framebgr, camara.cameraMatrix, camara.distCoeffs, None, matrix)
            framerecortado = framerectificado[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            arucoFound = findArucoMarkers(framerecortado)
            
            if len(arucoFound[0])>0:
                for corner, id in zip(arucoFound[0], arucoFound[1]):
                    framerecortado = augmentAruco(corner, id, framerecortado, lena)
                    getCenter(corner, id, framerecortado)
                    drawArucoAxis(corner, id ,framerecortado, camara.cameraMatrix, camara.distCoeffs, 9, 10)
                   

            cv2.imshow("RECORTADO", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                final = True
        else:
            final = True
else:
    print("No se pudo acceder a la cámara.")
