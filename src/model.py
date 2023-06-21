import cv2
import numpy as np
import settings

#Dado el modelo escaneado de una persona, construir el modelo de la camiseta
def getShirtModelPoints(lmList):
    
    #Obtener los tamaños según la talla actual
    size = settings.SIZE[settings.CURRENT_TALLA]
    half_size = size/2

    #Subimos los puntos de los hombros según la talla
    p1 = np.add(lmList[11][1:3], [0,-size]) 
    p2 = np.add(lmList[12][1:3], [0,-size])

    #Aquí calculamos si el usuario tiene el brazo levantado hallando la distancia entre el codo y el hombro
    diff1 = np.subtract(lmList[13][1:3],lmList[11][1:3])
    diff2 = np.subtract(lmList[14][1:3],lmList[12][1:3])

    p8 = np.array(p1 + 2/3*diff1+[size,0]).astype(int)
    p3 = np.array(p2 + 2/3*diff2+[-size,0]).astype(int)

    #Para las esquinas P4 y P7, fijamos un mínimo de 30 en Y por si el codo y el hombre están muy juntos
    p7 = np.add(p1, [0, np.absolute(3/4*diff1[1]) if np.absolute(diff1[1])>30 else 30 ]).astype(int)
    p4 = np.add(p2, [0, np.absolute(3/4*diff2[1]) if np.absolute(diff2[1])>30 else 30 ]).astype(int)
    
    #Las esquinas inferiores son un poco menos anchas que las caderas del modelo humano
    p6 = np.array([1/2*(lmList[23][1]+p1[0]), lmList[23][2]]).astype(int)
    p5 = np.array([1/2*(lmList[24][1]+p2[0]), lmList[24][2]]).astype(int)
    
    #Puntos finales para la camiseta en el orden para ser construido
    points = np.array([p2,p1,p8,p7,p6,p5,p4,p3], np.int32)

    return points


def getImageInShirt(points, img, frame_shape, show_resized_img=False, show_shirt_window=False):

    #Creamos una nueva ventana con fondo negro
    shirt_window = np.zeros(frame_shape, dtype=np.uint8)
    tratado = False

    #Determinar el nuevo tamaño de la imagen dado los puntos del modelo de camiseta
    min_w = np.min(points[:,0])  
    min_h = np.min(points[:,1])
    max_w = np.max(points[:,0])  
    max_h = np.max(points[:,1]) 

    '''Si la altura mínima o la anchura mínima de la camiseta sale fuera de la ventana, entonces no podemos obtener los valores de la matriz y realizar el AND'''
    if 0<=min_w and min_w<frame_shape[1] and 0<=min_h and min_h<frame_shape[0]:
        nuevo_w=max_w-min_w
        nuevo_h=max_h-min_h
        nuevo_dim = (nuevo_w, nuevo_h)

        resized_img = cv2.resize(img, nuevo_dim, interpolation=cv2.INTER_AREA)

        if show_resized_img:
            cv2.imshow("Resized image",resized_img)

        #Pintamos la camiseta de blanco
        cv2.fillConvexPoly(shirt_window, points, (255,255,255) )

        #Determinar el rectángulo que inscribe a la camiseta en la ventana, tomando desde los puntos mínimos de la camiseta hasta los mínimos entre los puntos máximos de la camiseta y los bordes de la ventana
        rectangulo = shirt_window[min_h:np.min([max_h, shirt_window.shape[0]] ), min_w:np.min([max_w, shirt_window.shape[1]] )]

        #Calculamos la región máxima de la imagen redimensionada acorde con el rectángulo anterior para hacer AND
        #Por ejemplo, si en el rectángulo se ha tomado en la primera coordenada min_h:max_h, en resized_img tomará 0:nuevo_h pues en ese caso max_h está dentro de la ventana
        limited_img = resized_img[0:np.min([shirt_window.shape[0]-min_h, nuevo_h ]), 0:np.min([shirt_window.shape[1]-min_w, nuevo_w]) ]

        #Hacer un AND del rectángulo con la imagen limitada
        shirt_window[min_h:np.min([max_h, shirt_window.shape[0]] ), min_w:np.min([max_w, shirt_window.shape[1]])]=cv2.bitwise_and(rectangulo, limited_img)
        
        if show_shirt_window:
            cv2.imshow("Shirt with image", shirt_window)

        tratado = True

    return (tratado, shirt_window)