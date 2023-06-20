import speech_recognition as sr
import pyttsx3 as tts
import settings 

def change_voice(engine):
    for voice in engine.getProperty('voices'):
        if "SPANISH" in voice.name.upper() or "ESPAÑOL" in voice.name.upper() or 'es_ES' in voice.languages:
            engine.setProperty('voice', voice.id)
            break

def recognizeCommand():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    speaker = tts.init()
    err_msg = "Lo siento, no he podido entenderte"
    speaker.setProperty('rate', 145)
    change_voice(speaker)

    while not settings.FINAL:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = recognizer.listen(source,phrase_time_limit=3)
                recognition = recognizer.recognize_google(audio, language="es-ES", show_all=True)
                response = "Unable to recognize speech"

                if len(recognition)>0:
                    print(recognition)
                    if recognition['alternative'][0]['confidence']>0.65:
                        response = recognition['alternative'][0]['transcript']
                        print(recognition['alternative'][0])
                    else:
                        speaker.say(err_msg)
                        speaker.runAndWait()

        except sr.RequestError:
            response = "API unavailable"
        except sr.UnknownValueError:
            response = "Unable to recognize speech"
        except sr.WaitTimeoutError:
            response = "Timeout"
        except ConnectionError:
            response = "Error de conexión"

        #print(response)