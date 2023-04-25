import cv2
import numpy as np
import sys
import os
if os.path.exists('camara.py'):
    import camara
else:
    print("Es necesario realizar la calibración de la cámara")
    exit()

lena = cv2.imread("../lena.tif")
DIC = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
parametros = cv2.aruco.DetectorParameters()

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

            (corners, ids, rejected) = cv2.aruco.detectMarkers(framerecortado, DIC, parameters=parametros)
            if len(corners)>0:
                #print(ids.flatten)
                for i in range(len(corners)):
                    

                    cv2.polylines(framerecortado, [corners[i].astype(int)], True, (0,255,0), 4)
                    centro = corners[i][0][0]
                    for j in range(3):
                        centro = centro + corners[i][0][j+1]
                    centro = centro / 4

                    dst_points = corners[i].reshape(4,2)
                    dst_points = dst_points.astype(int)

                    src_h, src_w = lena.shape[:2]
                    frame_h, frame_w = framerecortado.shape[:2]
                    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])
                    H, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)
                    warp_image = cv2.warpPerspective(lena, H, (frame_w, frame_h))
                    #cv2.imshow("warp image", warp_image)
                    cv2.fillConvexPoly(mask, dst_points, 255)
                    results = cv2.bitwise_and(warp_image, warp_image, framerecortado, mask=mask)



            cv2.imshow("RECORTADO", framerecortado)
            if cv2.waitKey(1) == ord(' '):
                final = True
        else:
            final = True
else:
    print("No se pudo acceder a la cámara.")
