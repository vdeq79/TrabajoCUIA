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
original = cv2.imread("../lena.tif")

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
                '''corner = []
                for i in [11,12,24,23]:
                    corner.append([lmList[i][1], lmList[i][2]])

                corner = np.array(corner)'''
                
                
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
                
                #points puntos finales para la camiseta
                points = np.array([p2,p1,p8,p7,p6,p5,p4,p3], np.int32)

                #for point in points:
                #    framerecortado = cv2.circle(framerecortado, point, radius=5, color=(255,0,0), thickness=-1)

                nuevo_w = p8[0]-p3[0]
                nuevo_h = p5[1]-p2[1]
                nuevo_dim = (nuevo_w, nuevo_h)


                tshirt = cv2.resize(original, nuevo_dim, interpolation=cv2.INTER_AREA)

                src_h, src_w = tshirt.shape[:2]

                src_points = np.array([[1/5*src_w,0], [4/5*src_w,0], [src_w, 1/3*src_h], [4/5*src_w, 1/3*src_h], [4/5*src_w, src_h], [1/5*src_w, src_h], [1/5*src_w, 1/3*src_h], [0,1/3*src_h]],np.int32)


                mask = np.zeros(tshirt.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [points], -1, (255,255,255), -1, cv2.LINE_AA)

                res = cv2.bitwise_or(tshirt, tshirt, mask=mask)
                cv2.imshow("Samed size black image", tshirt)
                dst_points = points


                #matrix2, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
                #warp_image = cv2.warpPerspective(res, matrix2, (framerecortado.shape[1], framerecortado.shape[0]))


                points = points.reshape(-1,1,2)

                cv2.polylines(framerecortado, points, True, (255,0,0), 8, cv2.LINE_AA)
                cv2.fillConvexPoly(framerecortado,dst_points, (0,0,0) )

                #nueva_ventana = framerecortado.copy()
                #nueva_ventana[nueva_ventana!=0] = 255
                nueva_ventana = np.zeros(framerecortado.shape, dtype=np.uint8)
                #print("1",p3[0],p8[0])
                #print("2",p2[1],p5[1])
                #print("3",p2[0],p5[0])
                #print(nueva_ventana.shape)

                #print([nueva_ventana.shape[1]-p2[1], nuevo_h ])
                cv2.fillConvexPoly(nueva_ventana,dst_points, (255,255,255) )
                rectangulo = nueva_ventana[p2[1]:np.min([p5[1], nueva_ventana.shape[0]] ), p3[0]:p8[0]]
                auxiliar = rectangulo + tshirt[0:np.min([nueva_ventana.shape[0]-p2[1], nuevo_h ]),]

                #usar and en lugar de sumar directamente
                nueva_ventana[p2[1]:np.min([p5[1], nueva_ventana.shape[0]] ), p3[0]:p8[0]]=cv2.bitwise_and(rectangulo, tshirt[0:np.min([nueva_ventana.shape[0]-p2[1], nuevo_h ]),])

                #nueva_ventana[p2[1]:np.min([p5[1], nueva_ventana.shape[0]] ), p3[0]:p8[0]] = [auxiliar[i] if np.any(rectangulo[i]==0) else rectangulo[i] for i in range(len(rectangulo))]

                '''for i in range(rectangulo.shape[0]):
                    for j in range(rectangulo.shape[1]):
                        if np.all(rectangulo[i,j]==0):
                            rectangulo[i,j]=auxiliar[i,j]'''
                
                

                #mask += warp_image
                cv2.imshow("mask", nueva_ventana)
                

                #cv2.imshow("warp", warp_image)
                #framerecortado+=res

                #framerecortado = cv2.bitwise_and(framerecortado, res )



            cv2.imshow("RECORTADO", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                final = True
        else:
            final = True
else:
    print("No se pudo acceder a la cámara.")
