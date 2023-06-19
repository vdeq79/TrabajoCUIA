import speech_recognition as sr
import time
import pyttsx3 as tts

def change_voice(engine):
    for voice in engine.getProperty('voices'):
        if "SPANISH" in voice.name.upper() or "ESPAÑOL" in voice.name.upper():
            engine.setProperty('voice', voice.id)
            return True


r = sr.Recognizer()
mic = sr.Microphone()
sp = tts.init()
sp.setProperty('rate', 145)
change_voice(sp)



while True:

    try:
        with mic as source:
            start_time = time.time()
            r.adjust_for_ambient_noise(source, duration=0.2)
            print("1 --- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            audio = r.listen(source,phrase_time_limit=3)
            print("2 --- %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            response = r.recognize_google(audio, language="es-ES", show_all=True)
            print("3 --- %s seconds ---" % (time.time() - start_time))
            sp.say(response['alternative'][0]['transcript'])
            sp.runAndWait()
            print(response['alternative'][0])

    except sr.RequestError:
        response = "API unavailable"
        r = sr.Recognizer()
        sp.say(response)
        sp.runAndWait()
        continue
    except sr.UnknownValueError:
        response = "Unable to recognize speech"
        sp.say(response)
        sp.runAndWait()
        continue
    except sr.WaitTimeoutError:
        response = "Timeout"
        sp.say(response)
        sp.runAndWait()
        continue
    except ConnectionError:
        response = "Error de conexión"
        sp.say(response)
        sp.runAndWait()
        r = sr.Recognizer()
        continue


