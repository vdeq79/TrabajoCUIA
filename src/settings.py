import os
import cv2
import numpy as np
import face_recognition as face
import json

global TALLA
TALLA = ['XS', 'S', 'M', 'L', 'XL']

#Variable para las tallas
global SIZE 
SIZE = {'XS':14, 'S': 20, 'M':26, 'L':32, 'XL':38}

#Directorio actual
global DIR_NAME
DIR_NAME = os.path.dirname(__file__)
global ROOT_DIR
ROOT_DIR= os.path.realpath(os.path.join(DIR_NAME, '..'))

GENERAL_USER_NAME = 'GENERAL_USER'

def init():
    #Se ha reconocido el usuario
    global USER_RECOGNIZED 
    USER_RECOGNIZED = False

    #Máximo intento de reconocimiento de usuario
    global MAX_TRIES
    MAX_TRIES = 100

    #Actual intento para reconocer al usuario
    global CURRENT_TRY 
    CURRENT_TRY = 0

    #Imagenes de usuarios cargados
    global USER_NAMES 
    USER_NAMES = []
    global USER_CODS
    USER_CODS = []

    USER_IMG_FOLDER = os.path.join(ROOT_DIR, 'usr' ,'img')

    for filename in os.listdir( USER_IMG_FOLDER ):
        #Utilizo esta función en caso de que la ruta contenga caracteres especiales
        img = cv2.imdecode(np.fromfile(os.path.join(USER_IMG_FOLDER, filename), dtype=np.uint8), cv2.IMREAD_UNCHANGED) 

        #Leer las imagenes de usuarios y obtener las codificaciones de sus caras
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            locs = face.face_locations(img_rgb)
            cod = face.face_encodings(img_rgb, locs, model='large')
            USER_NAMES.append(os.path.splitext(filename)[0])
            USER_CODS.append(cod)

    #Indica el fin del programa
    global FINAL
    FINAL = False

    #Talla general
    global CURRENT_TALLA 
    CURRENT_TALLA = 'M'

    #Usuario general
    global CURRENT_USER 
    CURRENT_USER = GENERAL_USER_NAME

def initImages():
    global IMAGES_NAMES
    global IMG_FOLDER
    global CURRENT_IMG_NAME

    CURRENT_IMG_NAME = None
    IMAGES_NAMES = []
    IMG_FOLDER = os.path.join(ROOT_DIR, 'img')

    for filename in os.listdir( IMG_FOLDER ):
        #Utilizo esta función en caso de que la ruta contenga caracteres especiales
        IMAGES_NAMES.append(filename) 
        
    CURRENT_IMG_NAME = IMAGES_NAMES[0]
    loadImage()

def loadUserPreferences():
    global USER_PREFERENCE
    global CURRENT_TALLA
    global CURRENT_IMG_NAME
    USER_PREFERENCE = {}

    PREFERENCE_FILE = os.path.join(ROOT_DIR, 'usr', 'preference.json')
    if(os.path.exists(PREFERENCE_FILE)):
        f = open(PREFERENCE_FILE)
        USER_PREFERENCE = json.load(f)
        
        if CURRENT_USER in USER_PREFERENCE:
            CURRENT_TALLA = USER_PREFERENCE[CURRENT_USER]['TALLA']
            CURRENT_IMG_NAME = USER_PREFERENCE[CURRENT_USER]['IMAGE']
            loadImage()
        else:
            USER_PREFERENCE[CURRENT_USER] = {'TALLA': CURRENT_TALLA, 'IMAGE': CURRENT_IMG_NAME}

        f.close()


def saveUserPreferences():
    if (CURRENT_USER!=GENERAL_USER_NAME):
        PREFERENCE_FILE = os.path.join(ROOT_DIR, 'usr', 'preference.json')
        USER_PREFERENCE[CURRENT_USER] = {'TALLA': CURRENT_TALLA, 'IMAGE': CURRENT_IMG_NAME}

        with open(PREFERENCE_FILE, 'w+') as outfile:
            json.dump(USER_PREFERENCE, outfile)

def loadImage():
    global CURRENT_IMG
    global CURRENT_IMG_NAME
    #Prevención de errores
    try:
        Index = IMAGES_NAMES.index(CURRENT_IMG_NAME)
    except ValueError:
        Index = 0
        CURRENT_IMG_NAME = IMAGES_NAMES[Index]

    for i in range(len(IMAGES_NAMES)):
        try:
            CURRENT_IMG = cv2.imdecode(np.fromfile(os.path.join(IMG_FOLDER, CURRENT_IMG_NAME), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except:
            Index = Index+1
            CURRENT_IMG_NAME = IMAGES_NAMES[Index]
            continue
        break
    