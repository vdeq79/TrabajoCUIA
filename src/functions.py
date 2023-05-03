import cv2
import numpy as np

def findArucoMarkers(img, markerSize=5, totalMarkers=250):
    key = getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = cv2.aruco.getPredefinedDictionary(key)
    parametros = cv2.aruco.DetectorParameters()

    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, arucoDict, parameters=parametros)

    return [corners, ids]

def augmentAruco(corner, id, img, imgAug):
    #Determinar la altura y anchura de la imagen
    frame_h, frame_w = img.shape[:2]

    #Identifico los 4 vértices del Aruco
    dst_points = corner.reshape(4,2)
    dst_points = dst_points.astype(int)

    #Determino la altura y anchura de la imagen que queremos sustituir
    src_h, src_w = imgAug.shape[:2]
    src_points = np.array([[0, 0], [src_w, 0], [src_w, src_h], [0, src_w]])

    #Matriz para la perspectiva
    matrix, _ = cv2.findHomography(srcPoints=src_points, dstPoints=dst_points)

    #Conseguimos la imagen sustituida en la posición del Aruco
    warp_image = cv2.warpPerspective(imgAug, matrix, (frame_w, frame_h))

    #Pintamos de negro en la imagen original en la posición del Aruco
    cv2.fillConvexPoly(img, dst_points, (0,0,0))

    #Superponemos las dos imágenes
    warp_image += img
    return warp_image

#Tamaño en metros
def drawArucoAxis(corner, id, img, cameraMatrix, distCoeffs, markerSize, axisSize):
    #Determinamos los vectores de rotación y de traslación
    rVec, tVec, _ = cv2.aruco.estimatePoseSingleMarkers(corner, markerSize, cameraMatrix, distCoeffs)
    #print(tVec)
    #print(corner)
    cv2.drawFrameAxes(img, cameraMatrix, distCoeffs, rVec, tVec, axisSize)
    cv2.putText(img, f"Dist:{tVec[0][0][2]}", corner[0][0].astype(int), cv2.FONT_HERSHEY_PLAIN, 1.3 ,(0,0,255), 2, cv2.LINE_AA)


def getCenter(corner, id, img, draw=True):
    centro = np.sum(corner)/4


