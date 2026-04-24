from vosk import Model, KaldiRecognizer
import sounddevice as sd
import queue
import json
import pyttsx3


class VoiceAssistant:
    def __init__(
        self,
        model_path="/home/tamim/Desktop/ELEC3442-Group-8/vosk-model-small-en-us-0.15",
        samplerate=None,
        blocksize=16000,
        channels=1,
        enable_tts=True
    ):
        self.model_path = model_path
        self.blocksize = blocksize
        self.channels = channels
        self.enable_tts = enable_tts

        self.q = queue.Queue()

        print("Loading Vosk model...")
        self.model = Model(self.model_path)
        print("Vosk model loaded.")

        device_info = sd.query_devices(kind="input")

        if samplerate is None:
            self.samplerate = int(device_info["default_samplerate"])
        else:
            self.samplerate = samplerate

        print("Using input device:", device_info["name"])
        print("Using sample rate:", self.samplerate)

        if self.enable_tts:
            self.engine = pyttsx3.init()
        else:
            self.engine = None

    def callback(self, indata, frames, time, status):
        if status:
            print("Audio status:", status)

        self.q.put(bytes(indata))

    def listen(self, seconds=3):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        print(f"Speak for {seconds} seconds ...")

        rec = KaldiRecognizer(self.model, self.samplerate)

        with sd.RawInputStream(
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            dtype="int16",
            channels=self.channels,
            callback=self.callback
        ):
            total_blocks = int(seconds * self.samplerate / self.blocksize)

            for _ in range(total_blocks):
                data = self.q.get()
                rec.AcceptWaveform(data)

        result = json.loads(rec.FinalResult())
        text = result.get("text", "")

        if text:
            print("You said:", text)
        else:
            print("You said: [no text recognized]")

        return text

    def speak(self, text):
        if not text:
            print("No text to speak.")
            return

        if not self.enable_tts or self.engine is None:
            print("TTS is disabled.")
            return

        self.engine.say(text)
        self.engine.runAndWait()

    def listen_and_repeat(self, seconds=3):
        text = self.listen(seconds=seconds)

        if text:
            self.speak(text)
        else:
            print("No recognizable speech.")

        return text
