import cv2
import numpy as np
import os
from cvzone.PoseModule import PoseDetector
import threading
import settings
from model import *
import speechRecognition as xsr
import faceRecognition as xface

if os.path.exists(os.path.join(settings.DIR_NAME, 'camara.py')):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()

cap = cv2.VideoCapture(0)
detector = PoseDetector()
#original = cv2.imdecode(np.fromfile(os.path.join(settings.ROOT_DIR, 'img', 'zebra.jpg'), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

settings.init()
settings.initImages()

xsr.sayMsg("Intentando detectar usuario, espere por favor")
thread = threading.Thread(target=xsr.recognizeCommand)
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
                    tratado, shirt_window = getImageInShirt(points, settings.IMAGES[settings.CURRENT_IMG_POS], framerecortado.shape)

                    if tratado:
                        #Pintamos la camiseta de negro
                        cv2.fillConvexPoly(framerecortado,points, (0,0,0) )
                        #Juntamos ambas ventanas
                        framerecortado = cv2.bitwise_or(framerecortado, shirt_window )

            if(settings.USER_RECOGNIZED):
                cv2.imshow("main", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                settings.FINAL = True
        else:
            settings.FINAL = True
else:
    print("No se pudo acceder a la cámara.")

thread.join()