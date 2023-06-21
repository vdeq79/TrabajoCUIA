import speech_recognition as sr
import pyttsx3 as tts
import settings 

def change_voice(engine):
    for voice in engine.getProperty('voices'):
        if "SPANISH" in voice.name.upper() or "ESPAÑOL" in voice.name.upper() or 'es_ES' in voice.languages:
            engine.setProperty('voice', voice.id)
            break

recognizer = sr.Recognizer()
mic = sr.Microphone()
speaker = tts.init()
err_msg = "Lo siento, no he podido entenderte"
speaker.setProperty('rate', 145)
change_voice(speaker)

AGRANDAR = ['agrandar', 'más grande', 'grande']
ACHICAR = ['achicar', 'más pequeño', 'pequeño']
CAMBIAR = ['cambiar']
SALIR = ['salir']

def AGRANDAR_TALLA():
    index = settings.TALLA.index(settings.CURRENT_TALLA)
    if(index+1<len(settings.TALLA)):
        settings.CURRENT_TALLA = settings.TALLA[index+1]
    print(settings.CURRENT_TALLA)    
    sayMsg("Ejecutado el comando agrandar")

def ACHICAR_TALLA():
    index = settings.TALLA.index(settings.CURRENT_TALLA)
    if(0<=index-1):
        settings.CURRENT_TALLA = settings.TALLA[index-1]  
    print(settings.CURRENT_TALLA)    
    sayMsg("Ejecutado el comando achicar")

def CAMBIAR_IMG():
    settings.CURRENT_IMG_POS = (settings.CURRENT_IMG_POS+1)%len(settings.IMAGES)
    print(settings.CURRENT_IMG_POS)
    sayMsg("Ejecutado el cambio de imagen")

def SALIR_GUARDAR():
    sayMsg("Saliendo del programa")
    settings.FINAL = True
    print(settings.FINAL)
    #Grabar las preferencias

COMMANDS = {AGRANDAR_TALLA: AGRANDAR, ACHICAR_TALLA: ACHICAR, CAMBIAR_IMG:CAMBIAR, SALIR_GUARDAR:SALIR}

def executeCommand(order):
    for func in COMMANDS:
        if any(command in order for command in COMMANDS[func]):
            func()
            break

def recognizeCommand():
    while not settings.FINAL:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = recognizer.listen(source,phrase_time_limit=5)
                recognition = recognizer.recognize_google(audio, language="es-ES", show_all=True)
                response = "Unable to recognize speech"

                if len(recognition)>0:
                    print(recognition)
                    if recognition['alternative'][0]['confidence']>0.65:
                        response = recognition['alternative'][0]['transcript']
                        print(recognition['alternative'][0])
                    '''else:
                        speaker.say(err_msg)
                        speaker.runAndWait()'''

        except sr.RequestError:
            response = "API unavailable"
        except sr.UnknownValueError:
            response = "Unable to recognize speech"
        except sr.WaitTimeoutError:
            response = "Timeout"
        except ConnectionError:
            response = "Error de conexión"

        executeCommand(response)


def sayMsg(msg):
    speaker.say(msg)
    speaker.runAndWait()
