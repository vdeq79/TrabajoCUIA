import os
import cv2
import face_recognition as face

#Variable para las tallas
SIZE = {'XS':8, 'S': 12, 'M':16, 'L':20, 'XL':24}

#Directorio actual
DIR_NAME = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(DIR_NAME, '..'))

#Se ha reconocido el usuario
USER_RECOGNIZED = False

#Máximo intento de reconocimiento de usuario
MAX_TRIES = 100

#Actual intento para reconocer al usuario
CURRENT_TRY = 0

#Imagenes de usuarios cargados
USER_NAMES = []
USER_CODS = []
USER_IMG_FOLDER = os.path.join(ROOT_DIR, 'usr' ,'img')

for filename in os.listdir( USER_IMG_FOLDER ):
    img = cv2.imread(os.path.join(USER_IMG_FOLDER, filename))
    if img is not None:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locs = face.face_locations(img_rgb)
        cod = face.face_encodings(img_rgb, locs, model='large')
        USER_NAMES.append(os.path.splitext(filename)[0])
        USER_CODS.append(cod)

FINAL = False
TALLA = 'M'
CURRENT_USER = 'GENERAL_USER'
