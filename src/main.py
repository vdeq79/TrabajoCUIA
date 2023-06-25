import cv2
import numpy as np
import os
from cvzone.PoseModule import PoseDetector
import threading
import settings
from model import *
import speechRecognition as xsr
import faceRecognition as xface

camara_imported=True

if os.path.exists(os.path.join(settings.DIR_NAME, 'camara.py')):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    camara_imported = False

cap = cv2.VideoCapture(0)
detector = PoseDetector()

settings.init()
settings.initImages()

xsr.sayMsg("Intentando detectar usuario, espere por favor")
thread = threading.Thread(target=xsr.recognizeCommand)
thread.start()

if cap.isOpened():
    hframe = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wframe = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("Tamaño del frame de la cámara: ", wframe, "x", hframe)

    if camara_imported:
        matrix, roi = cv2.getOptimalNewCameraMatrix(camara.cameraMatrix, camara.distCoeffs, (wframe,hframe), 1, (wframe,hframe))
        roi_x, roi_y, roi_w, roi_h = roi

    while not settings.FINAL:
        ret, framebgr = cap.read()

        if ret:
            if camara_imported:
                # Aquí procesamos el frame
                framerectificado = cv2.undistort(framebgr, camara.cameraMatrix, camara.distCoeffs, None, matrix)
                framerecortado = framerectificado[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
            else:
                framerecortado = framebgr

            if not settings.USER_RECOGNIZED and settings.CURRENT_TRY<settings.MAX_TRIES:
                framerecortado = xface.recognizeUser(framerecortado)
            else:
                if not settings.USER_RECOGNIZED:
                    xsr.sayMsg("Bienvenido")
                    settings.USER_RECOGNIZED = True

                framerecortado = detector.findPose(framerecortado, draw=False)
                lmList, bboxInfo = detector.findPosition(framerecortado, draw=False)

                if bboxInfo:
                    points = getShirtModelPoints(lmList)
                    
                    tratado, shirt_window = getImageInShirt(points, settings.CURRENT_IMG, framerecortado.shape)

                    if tratado:
                        #Pintamos la camiseta de negro
                        cv2.drawContours(framerecortado, np.array([points]), -1, (0,0,0), thickness=-1 )
                        #Juntamos ambas ventanas
                        framerecortado = cv2.bitwise_or(framerecortado, shirt_window )

            if(settings.USER_RECOGNIZED):
                #framerecortado = cv2.resize(framerecortado, (1500,1080), interpolation=cv2.INTER_AREA)
                cv2.imshow("main", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                settings.FINAL = True
        else:
            settings.FINAL = True
else:
    print("No se pudo acceder a la cámara.")

settings.saveUserPreferences()
thread.join()