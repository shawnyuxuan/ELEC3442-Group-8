import speech_recognition as sr
import requests
import pyttsx3
import argparse
import sys
import os
import threading
import queue
import platform
import subprocess
import time
from flask import Flask, request, jsonify
from src.speech_to_text import VoiceAssistant, build_voice_feedback_payload, submit_voice_feedback
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)

# Queue for handling TTS requests so they don't block the API
tts_queue = queue.Queue()

# Global config
SERVER_URL = os.getenv("VOICE_SERVER_URL", "http://192.168.1.100:5888/voice")
ENGINE = None


def play_wake_ack() -> None:
    """Play a short non-blocking acknowledgement sound after wake-word detection."""
    # Terminal bell fallback.
    try:
        print("\a", end="", flush=True)
    except Exception:
        pass

    # macOS system sound if available.
    if platform.system().lower() == "darwin":
        sound_path = "/System/Library/Sounds/Glass.aiff"
        if os.path.exists(sound_path):
            try:
                subprocess.Popen(
                    ["afplay", sound_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

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

def tts_worker(engine):
    """Worker thread that consumes texts from the queue and speaks them out loud."""
    print("[TTS Worker] Started")
    while True:
        text = tts_queue.get()
        if text is None:
            break
        print(f"\n[TTS Playing]: {text}")
        if engine:
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS Error] {e}")
                # Fallback to espeak
                import subprocess
                subprocess.run(["espeak", text])
        else:
            # Fallback to espeak
            os.system(f'espeak "{text}"')
        tts_queue.task_done()

@app.route('/speak', methods=['POST'])
def handle_speak_request():
    """Endpoint for Pi 1 to trigger speech on Pi 2."""
    payload = request.get_json(silent=True)
    if not payload or "text" not in payload:
        return jsonify({"status": "error", "message": "Missing 'text' in payload"}), 400
        
    text = payload["text"]
    tts_queue.put(text)
    return jsonify({"status": "queued"}), 202

def mic_worker_thread(server_url):
    """
    Background worker that continuously listens for the wake word using VoiceAssistant,
    then captures a command and sends it to the main Pi 1 server.
    """
    print("[Mic Worker] Started listening for wake words...")
    wake_words = tuple(token.strip().lower() for token in os.getenv("VOICE_WAKE_WORDS", "hey calendar").split(",") if token.strip())
    
    is_local_stt = os.getenv("VOICE_STT_LOCAL", "false").lower() == "true"
    language = os.getenv("VOICE_LANGUAGE", "en-US")
    model_path = os.getenv("VOICE_MODEL_PATH", os.path.join(os.getcwd(), "vosk-model-small-en-us-0.15"))
    trigger_seconds = max(1, int(os.getenv("VOICE_TRIGGER_LISTEN_SECONDS", "1")))
    command_seconds = max(3, int(os.getenv("VOICE_COMMAND_LISTEN_SECONDS", "12")))
    timeout_threshold = max(1, int(os.getenv("VOICE_TIMEOUT_THRESHOLD", "5")))
    post_wake_delay_ms = max(0, int(os.getenv("VOICE_POST_WAKE_DELAY_MS", "350")))
    min_command_words = max(1, int(os.getenv("VOICE_MIN_COMMAND_WORDS", "3")))
    max_command_retries = max(0, int(os.getenv("VOICE_COMMAND_MAX_RETRIES", "1")))
    local_samplerate_env = os.getenv("VOICE_LOCAL_SAMPLE_RATE", "").strip()
    local_blocksize = max(1024, int(os.getenv("VOICE_LOCAL_BLOCKSIZE", "4000")))

    local_samplerate = None
    if local_samplerate_env:
        try:
            local_samplerate = int(local_samplerate_env)
        except ValueError:
            print(f"[Mic Worker] invalid VOICE_LOCAL_SAMPLE_RATE='{local_samplerate_env}', ignore it")

    try:
        assistant = VoiceAssistant(
            model_path=model_path,
            enable_tts=False,
            is_local=is_local_stt,
            language=language,
            samplerate=local_samplerate,
            blocksize=local_blocksize,
        )
    except Exception as exc:
        print(f"[Mic Worker] failed to initialize microphone listener: {exc}")
        return

    print(f"[Mic Worker] listening for wake words: {wake_words}")

    def send_feedback(payload: dict):
        transcript = payload.get("transcript", "")
        print(f"[Mic Worker] Sending payload to {server_url}: {transcript}")
        try:
            data = submit_voice_feedback(payload=payload, feedback_url=server_url, timeout=20)
            response_text = data.get("response")
            if response_text:
                tts_queue.put(response_text)
        except requests.exceptions.RequestException as e:
            print(f"[Mic Worker] Connection failed: {e}")

    def on_feedback_captured(payload: dict):
        # Avoid blocking the microphone loop on network latency.
        threading.Thread(target=send_feedback, args=(payload,), daemon=True).start()

    # Listen loop
    while True:
        heard = assistant.listen(seconds=trigger_seconds, timeout_threshold=timeout_threshold)
        normalized = heard.strip().lower()
        if not normalized:
            continue

        if not any(wake_word in normalized for wake_word in wake_words):
            continue

        print(f"\n[Mic Worker] Wake word detected: '{heard}'. Listening for command...")
        play_wake_ack()
        
        if post_wake_delay_ms > 0:
            time.sleep(post_wake_delay_ms / 1000.0)

        transcript = ""
        for attempt in range(max_command_retries + 1):
            transcript = assistant.listen(seconds=command_seconds, timeout_threshold=timeout_threshold).strip()
            if transcript and len(transcript.split()) >= min_command_words:
                break

            if attempt < max_command_retries:
                print(
                    f"[Mic Worker] Command too short/unclear ('{transcript}'). "
                    f"Retrying capture {attempt + 1}/{max_command_retries}..."
                )
                play_wake_ack()

        if not transcript:
            print("[Mic Worker] No command transcript captured after wake word.")
            continue

        payload = build_voice_feedback_payload(
            transcript=transcript,
            date="today",
            source="microphone-wake-word",
            language=language,
            context={"wake_word": heard, "retries": max_command_retries},
        )
        on_feedback_captured(payload)

def main():
    global SERVER_URL, ENGINE
    
    parser = argparse.ArgumentParser(description="Voice Agent (Pi 2) - Two-Way Pipeline")
    parser.add_argument("--server", default="http://192.168.1.100:5888/voice", help="Pi 1 Server Voice URL")
    parser.add_argument("--port", type=int, default=5890, help="Local port for the TTS API")
    args = parser.parse_args()
    
    SERVER_URL = args.server
    
    # Initialize TTS engine
    try:
        ENGINE = pyttsx3.init()
        ENGINE.setProperty("rate", 150)
    except Exception as e:
        print(f"Warning: pyttsx3 initialization failed: {e}. Will fallback to espeak.")

    # Start TTS worker thread
    threading.Thread(target=tts_worker, args=(ENGINE,), daemon=True).start()

    # Start Mic STT worker thread if mic is available
    if check_microphone():
        threading.Thread(target=mic_worker_thread, args=(SERVER_URL,), daemon=True).start()
    else:
        print("Starting without microphone. (TTS-only node)")

    # Run the Flask API
    app.run(host="0.0.0.0", port=args.port, threaded=True)

if __name__ == "__main__":
    main()
