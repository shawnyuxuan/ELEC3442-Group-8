import datetime
import json
import queue
from typing import Callable

import pyttsx3
import requests
import sounddevice as sd
import speech_recognition as sr  # The online approach for voice recognition.
from vosk import KaldiRecognizer, Model


def build_voice_feedback_payload(
    transcript: str,
    date: str | None = None,
    source: str = "microphone",
    language: str = "en-US",
    context: dict | None = None,
) -> dict:
    normalized_transcript = str(transcript or "").strip()
    if not normalized_transcript:
        raise ValueError("transcript is required")

    payload = {
        "date": date or datetime.date.today().isoformat(),
        "transcript": normalized_transcript,
        "source": source,
        "language": language,
        "captured_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if context is not None:
        payload["context"] = context
    return payload


def submit_voice_feedback(payload: dict, feedback_url: str, timeout: int = 10) -> dict:
    response = requests.post(feedback_url, json=payload, timeout=timeout)
    response.raise_for_status()
    if not response.content:
        return {"status": "success"}
    return response.json()


class VoiceAssistant:
    def __init__(
        self,
        model_path,
        is_local=True,
        samplerate=None,
        blocksize=16000,
        channels=1,
        enable_tts=True,
        language="en-US"
    ):
        self.model_path = model_path
        self.blocksize = blocksize
        self.channels = channels
        self.enable_tts = enable_tts
        self.is_local = is_local
        
        self.q = queue.Queue()
        if self.is_local:
            print("Loading Vosk model...")
            self.model = Model(self.model_path)
            print("Vosk model loaded.")
        else:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.language = language
            print("Using online speech recognition with language:", self.language)
            print("--- Calibrating microphone for ambient noise ---")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            print("--- Calibration complete. ---")
                

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

    def listen(self, seconds=3, timeout_threshold=5):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

        if self.is_local:
            print(f"Listening for {seconds} seconds...")
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
                
        else:
            try:
                print(f"Speak within {timeout_threshold} seconds...")
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=timeout_threshold)
                text = self.recognizer.recognize_google(audio, language=self.language)
                print("You said:", text)
                return text
            except sr.WaitTimeoutError:
                print("Listening timed out while waiting for phrase to start")
                return ""
            except sr.UnknownValueError:
                print("Could not understand audio")
                return ""
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                return ""
            

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

    def listen_and_serialize_feedback(
        self,
        seconds=3,
        timeout_threshold=5,
        date: str | None = None,
        source: str = "microphone",
        context: dict | None = None,
        on_feedback: Callable[[dict], None] | None = None,
    ) -> dict:
        transcript = self.listen(seconds=seconds, timeout_threshold=timeout_threshold)
        if not transcript:
            return {}

        payload = build_voice_feedback_payload(
            transcript=transcript,
            date=date,
            source=source,
            language=getattr(self, "language", "en-US"),
            context=context,
        )

        if on_feedback is not None:
            on_feedback(payload)

        return payload
