from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import pyttsx3

q = queue.Queue()

model = Model("/home/tamim/Desktop/ELEC3442-Group-8/vosk-model-small-en-us-0.15")

device_info = sd.query_devices(kind = 'input')
samplerate =int(device_info['default_samplerate'])

def callback(indata, frames, time, status):
	if status:
		print("Audio status:",status)
	q.put(bytes(indata))

with sd.RawInputStream(samplerate=samplerate, blocksize=16000, dtype='int16', channels=1, callback=callback):
    print("Speak for 3 second ...")
    rec = KaldiRecognizer(model, samplerate)
    
    for _ in range( int(5 * 16000 / 8000)): #3 second
        data = q.get()
        if rec.AcceptWaveform(data):
            break
    
    result = json.loads(rec.FinalResult())
        
    print("You said:", result.get("text", "[no text recognized]")) 






