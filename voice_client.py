import speech_recognition as sr
import requests
import pyttsx3
import argparse
import sys
import os
from src.speech_to_text import build_voice_feedback_payload, submit_voice_feedback

def check_microphone():
    """Ensure a microphone is available for SpeechRecognition."""
    try:
        mics = sr.Microphone.list_microphone_names()
        if not mics:
            print("No microphones detected!")
            return False
        return True
    except Exception as e:
        print(f"Error accessing microphone: {e}")
        return False

def listen_and_recognize(recognizer, mic, language="en-US"):
    with mic as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("\n\n=== Listening! Speak now ===")
        try:
            # Capture audio 
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Processing speech...")
            # Use Google Web Speech API (free requires internet)
            text = recognizer.recognize_google(audio, language=language)
            print(f"Recognized text: {text}")
            return text
        except sr.WaitTimeoutError:
            print("Timed out waiting for speech.")
            return None
        except sr.UnknownValueError:
            print("Could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"Failed STT request: {e}")
            return None

def send_to_server(text, server_url):
    print(f"Sending text to server: {server_url}")
    try:
        payload = build_voice_feedback_payload(
            transcript=text,
            source="voice-client",
            language="en-US",
        )
        data = submit_voice_feedback(payload=payload, feedback_url=server_url, timeout=20)
        return data.get("response", "Error: No response key in JSON.")
    except requests.exceptions.RequestException as e:
        print(f"Connection failed: {e}")
        return None

def text_to_speech(text, engine):
    print(f"\nSpeaking: {text}")
    print("="*40)
    engine.say(text)
    engine.runAndWait()

def main():
    parser = argparse.ArgumentParser(description="Voice STT -> HTTP STT Pipeline")
    parser.add_argument("--server", default="http://192.168.1.100:5888/voice", help="Pi 1 Server URL")
    args = parser.parse_args()
    
    server_url = args.server
    
    if not check_microphone():
        print("Cannot start without microphone. Make sure sound device is connected.")
        sys.exit(1)
        
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Initialize TTS engine
    # on Pi, if pyttsx3 fails, fall back to espeak using os.system
    engine = None
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150) # Speech speed
    except Exception as e:
        print(f"Warning: TTS initialization failed: {e}. Will fallback to basic print/espeak.")

    while True:
        try:
            print("\nPress Enter to start recording (or type 'quit' to exit).")
            user_input = input("> ").strip().lower()
            if user_input in ["q", "quit", "exit"]:
                break
                
            text = listen_and_recognize(recognizer, mic)
            if not text:
                continue
                
            response_text = send_to_server(text, server_url)
            if response_text:
                if engine:
                    text_to_speech(response_text, engine)
                else:
                    # Fallback to espeak
                    print(f"Server replied: {response_text}")
                    os.system(f'espeak "{response_text}"')
                    
        except KeyboardInterrupt:
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()
