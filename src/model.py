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


#Dado el modelo escaneado de una persona, construir el modelo de la camiseta
def getShirtModelPoints(lmList):
    p1 = np.add(lmList[11][1:3], [0,-15]) 
    p2 = np.add(lmList[12][1:3], [0,-15])

    #Aquí calculamos si el usuario tiene el brazo levantado hallando la distancia entre el codo y el hombro
    diff1 = np.subtract(lmList[13][1:3],lmList[11][1:3])
    diff2 = np.subtract(lmList[14][1:3],lmList[12][1:3])

    p8 = np.array(p1 + 2/3*diff1+[16,-8]).astype(int)
    p3 = np.array(p2 + 2/3*diff2+[-16,-8]).astype(int)

    #Para las esquinas P4 y P7, fijamos un mínimo de 30 en Y por si el codo y el hombre están muy juntos
    p7 = np.add(p1, [0, np.absolute(3/4*diff1[1]) if np.absolute(diff1[1])>30 else 30 ]).astype(int)
    p4 = np.add(p2, [0, np.absolute(3/4*diff2[1]) if np.absolute(diff2[1])>30 else 30 ]).astype(int)
    
    #Las esquinas inferiores son un poco menos anchas que las caderas del modelo humano
    p6 = np.array([1/2*(lmList[23][1]+p1[0]), lmList[23][2]]).astype(int)
    p5 = np.array([1/2*(lmList[24][1]+p2[0]), lmList[24][2]]).astype(int)
    
    #points puntos finales para la camiseta
    points = np.array([p2,p1,p8,p7,p6,p5,p4,p3], np.int32)

    return points


def getImageInShirt(points, img, frame_shape):

    #Creamos una nueva ventana con fondo negro
    nueva_ventana = np.zeros(frame_shape, dtype=np.uint8)
    tratado = False

    #Determinar el nuevo tamaño de la imagen dado los puntos del modelo de camiseta
    min_w = np.min(points[:,0])  
    min_h = np.min(points[:,1])
    max_w = np.max(points[:,0])  
    max_h = np.max(points[:,1]) 

    if 0<=min_w and min_w<frame_shape[1] and 0<=min_h and min_h<frame_shape[0]:
        nuevo_w=max_w-min_w
        nuevo_h=max_h-min_h
        nuevo_dim = (nuevo_w, nuevo_h)

        resized_img = cv2.resize(img, nuevo_dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Resize",resized_img)

        #Pintamos la camiseta de blanco
        cv2.fillConvexPoly(nueva_ventana,points, (255,255,255) )

        #Determinar el rectángulo que inscribe a la camiseta
        rectangulo = nueva_ventana[min_h:np.min([max_h, nueva_ventana.shape[0]] ), min_w:np.min([max_w, nueva_ventana.shape[1]] )]

        #Calculamos la región máxima que podemos hacer and, pues es posible que algún punto salga fuera de la pantalla
        limited_img = resized_img[0:np.min([nueva_ventana.shape[0]-min_h, nuevo_h ]), 0:np.min([nueva_ventana.shape[1]-min_w, nuevo_w]) ]

        #Hacer un and del rectángulo con la imagen escalada
        nueva_ventana[min_h:np.min([max_h, nueva_ventana.shape[0]] ), min_w:np.min([max_w, nueva_ventana.shape[1]])]=cv2.bitwise_and(rectangulo, limited_img)
        
        cv2.imshow("mask", nueva_ventana)
        tratado = True

    return (tratado,nueva_ventana)