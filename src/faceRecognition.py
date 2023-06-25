import face_recognition as face
import cv2 
import settings
import speechRecognition as xsr

def recognizeUser(frame):

    miniframe = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None, fx=0.25, fy=0.25)
    locs = face.face_locations(miniframe)
    cods = face.face_encodings(miniframe, locs, model='large')
    if locs is not None:
        for i in range(len(locs)):

            comparison = [face.compare_faces(userCod, cods[i]) for userCod in settings.USER_CODS]            
            try:
                index = comparison.index([True])
            except ValueError:
                index = -1
                continue

            if index>=0:
                color = (0, 255, 0)
                settings.USER_RECOGNIZED = True
                settings.CURRENT_USER = settings.USER_NAMES[index]
                xsr.sayMsg("Bienvenido "+settings.CURRENT_USER)
                settings.loadUserPreferences()
                break
            else:
                color = (0, 0, 255)

            t, r, b, l = locs[i]
            cv2.rectangle(frame, (l*4,t*4), (r*4,b*4), color, 2)

    settings.CURRENT_TRY+=1
    return(frame)
