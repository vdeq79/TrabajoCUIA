import cv2
import numpy as np
import os
from cvzone.PoseModule import PoseDetector
import threading
import settings
from model import *
from speechRecognition import * 
from faceRecognition import *

if os.path.exists(os.path.join(settings.DIR_NAME, 'camara.py')):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()

cap = cv2.VideoCapture(0)
detector = PoseDetector()
original = cv2.imread(os.path.join(settings.ROOT_DIR, 'img', 'zebra.jpg'))

thread = threading.Thread(target=recognizeCommand)
thread.start()

if cap.isOpened():
    hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Tamaño del frame de la cámara: ", wframe, "x", hframe)

    matrix, roi = cv2.getOptimalNewCameraMatrix(camara.cameraMatrix, camara.distCoeffs, (wframe,hframe), 1, (wframe,hframe))
    roi_x, roi_y, roi_w, roi_h = roi

    while not settings.FINAL:
        ret, framebgr = cap.read()

        if ret:
            # Aquí procesamos el frame
            framerectificado = cv2.undistort(framebgr, camara.cameraMatrix, camara.distCoeffs, None, matrix)
            framerecortado = framerectificado[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            #response = recognizeCommand(r, mic, sp, err_msg)

            if not settings.USER_RECOGNIZED and settings.CURRENT_TRY<settings.MAX_TRIES:
                framerecortado = recognizeUser(framerecortado)
            else:
                framerecortado = detector.findPose(framerecortado, draw=False)
                lmList, bboxInfo = detector.findPosition(framerecortado, draw=False)

                if bboxInfo:
                    points = getShirtModelPoints(lmList)
                    tratado, shirt_window = getImageInShirt(points, original, framerecortado.shape)

                    if tratado:
                        #Pintamos la camiseta de negro
                        cv2.fillConvexPoly(framerecortado,points, (0,0,0) )
                        #Juntamos ambas ventanas
                        framerecortado = cv2.bitwise_or(framerecortado, shirt_window )


            cv2.imshow("RECORTADO", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                settings.FINAL = True
        else:
            settings.FINAL = True
else:
    print("No se pudo acceder a la cámara.")

thread.join()