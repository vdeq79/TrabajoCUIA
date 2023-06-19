import cv2
import numpy as np
import os
from cvzone.PoseModule import PoseDetector
from model import *


if os.path.exists('camara.py'):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()

cap = cv2.VideoCapture(0)
detector = PoseDetector()
original = cv2.imread("../zebra.jpg")

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

            framerecortado = detector.findPose(framerecortado, draw=False)
            lmList, bboxInfo = detector.findPosition(framerecortado, draw=False)

            if bboxInfo:

                points = getShirtModelPoints(lmList)
                tratado, nueva_ventana = getImageInShirt(points, original, framerecortado.shape)

                if tratado:
                    #Pintamos la camiseta de negro
                    cv2.fillConvexPoly(framerecortado,points, (0,0,0) )
                    #Juntamos ambas ventanas
                    framerecortado = cv2.bitwise_or(framerecortado, nueva_ventana )

            cv2.imshow("RECORTADO", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                final = True
        else:
            final = True
else:
    print("No se pudo acceder a la cámara.")
