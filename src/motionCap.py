import cv2
import numpy as np
import os
from cvzone.PoseModule import PoseDetector
from functions import *


if os.path.exists('camara.py'):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()

cap = cv2.VideoCapture(0)
detector = PoseDetector()
tshirt = cv2.imread("../lena.tif")

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
                corner = []
                for i in [11,12,24,23]:
                    corner.append([lmList[i][1], lmList[i][2]])

                corner = np.array(corner)
                
                
                p1 = np.add(lmList[11][1:3], [0,-15]) 
                p2 = np.add(lmList[12][1:3], [0,-15])

                diff1 = np.subtract(lmList[13][1:3],lmList[11][1:3])
                diff2 = np.subtract(lmList[14][1:3],lmList[12][1:3])

                p8 = np.array(p1 + 2/3*diff1+[16,-8]).astype(int)
                p3 = np.array(p2 + 2/3*diff2+[-16,-8]).astype(int)

                #print(diff1[1])

                p7 = np.add(p1, [0, np.absolute(3/4*diff1[1]) if np.absolute(diff1[1])>30 else 30 ]).astype(int)
                p4 = np.add(p2, [0, np.absolute(3/4*diff2[1]) if np.absolute(diff2[1])>30 else 30 ]).astype(int)
                
                p6 = np.array([1/2*(lmList[23][1]+p1[0]), lmList[23][2]]).astype(int)
                p5 = np.array([1/2*(lmList[24][1]+p2[0]), lmList[24][2]]).astype(int)
                

                #points = np.array([p1,p2,p3,p4,p5,p6,p7,p8], np.int32)

                points = np.array([p2,p1,p8,p7,p6,p5,p4,p3], np.int32)

                #for point in points:
                #    framerecortado = cv2.circle(framerecortado, point, radius=5, color=(255,0,0), thickness=-1)

                src_h, src_w = tshirt.shape[:2]
                #src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])
                #dst_points = np.array([p2,p1,p6,p5],int) 


                src_points = np.array([[1/5*src_w,0], [4/5*src_w,0], [src_w, 1/3*src_h], [4/5*src_w, 1/3*src_h], [4/5*src_w, src_h], [1/5*src_w, src_h], [1/5*src_w, 1/3*src_w], [0,1/3*src_w]],np.int32)
                dst_points = points
                #print(dst_points)

                matrix2, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
                warp_image = cv2.warpPerspective(tshirt, matrix2, (framerecortado.shape[1], framerecortado.shape[0]))

                points = points.reshape(-1,1,2)
                #cv2.fillPoly(framerecortado, [points], color=(255,255,255))


                cv2.fillConvexPoly(framerecortado,dst_points, (0,0,0) )
                #cv2.imshow("warp", warp_image)

                framerecortado+=warp_image


            cv2.imshow("RECORTADO", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                final = True
        else:
            final = True
else:
    print("No se pudo acceder a la cámara.")
