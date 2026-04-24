# Voice Assistant Integration (Multi-Pi Setup)

This document explains the setup and usage of the two-Pi voice assistant system for the Smart Calendar Agent. Since the main Raspberry Pi (Pi 1) operates headless (CLI only) with sensors, a secondary Raspberry Pi (Pi 2) is used exclusively for handling voice input (Speech-to-Text) and audio output (Text-to-Speech).

## Architecture Overview

The system is split into a **Frontend (Pi 2)** and a **Backend (Pi 1)** using a two-way HTTP communication pipeline:

1. **Pi 2 (Ears & Mouth)**: Runs `voice_agent.py`. It continuously listens for a wake word (e.g., "hey calendar") using a local microphone, captures the command, converts the audio to text using a local VOSK STT model or Google Web Speech API, and sends the text to Pi 1 via HTTP POST. It also runs a local Flask server on port `5890` with a `/speak` endpoint to receive text from Pi 1 and play it through a connected speaker (TTS) asynchronously.
2. **Pi 1 (The Brain)**: Runs the main `calendar-agent.py`. It receives text queries via Flask APIs (`/voice` and `/feedback` endpoints). It integrates current SenseHat sensor data, sleep quality, and calendar events to prompt the LLM. After performing calendar operations (like moving or creating events), Pi 1 sends an HTTP POST request to Pi 2's `/speak` endpoint to trigger a vocal confirmation or response.

## Hardware Requirements

* **Pi 1 (Backend)**: Raspberry Pi with SenseHat, internet connection.
* **Pi 2 (Frontend)**: Raspberry Pi with a USB/Bluetooth microphone and a speaker/headphone jack connected. Internet connection.

---

## 1. Setup Pi 1 (Backend / Main Agent)

Pi 1 hosts the core application and the API endpoint listening on port `5888`.

### Installation
Pi 1 requires the standard project dependencies.
```bash
# Clone the repository on Pi 1
cd ELEC3442-Group-8
pip install -r requirements.txt
```

### Running the Backend
Ensure you have your LLM API keys exported in your environment (e.g., `LLM_API_KEY`). You must also tell Pi 1 where the Pi 2 Voice Agent lives so it can send TTS requests back:

```bash
export PI2_VOICE_AGENT_URL="http://<PI_2_IP>:5890/speak"
python3 calendar-agent.py
```
*Note down the IP address of Pi 1 on your local network (e.g., `192.168.1.100`). You can find this by running `hostname -I` in a separate terminal on Pi 1.*

---

## 2. Setup Pi 2 (Frontend / Voice Agent)

Pi 2 needs `voice_agent.py`, the `src/speech_to_text.py` file, and the audio processing/Flask libraries.

### Installation

First, install the system-level audio dependencies (flac, pyaudio, and espeak as a TTS fallback):
```bash
sudo apt-get update
sudo apt-get install flac espeak python3-pyaudio
```

Next, install the Python libraries for recording, STT, web server endpoints, and HTTP requests. You can just clone the full repo or copy the needed files.
```bash
pip install Flask SpeechRecognition pyaudio pyttsx3 requests vosk
```

### File Transfer
Ensure `voice_agent.py` and the `src/` directory containing `speech_to_text.py` are present on Pi 2. If using local VOSK STT, download and extract the vosk-model-small-en-us-0.15 folder.

---

## 3. Usage & Interaction

With **Pi 1** already running `calendar-agent.py`:

1. Log into **Pi 2**.
2. Run the voice agent, pointing it to Pi 1's IP Address (replace `192.168.1.100` with the actual IP of Pi 1).

```bash
python3 voice_agent.py --server http://192.168.1.100:5888/voice --port 5890
```

3. The script will automatically listen for your wake word (e.g., `"hey calendar"`).
4. **Speak your request!** Just say "hey calendar" followed by your command (e.g., *"Hey calendar, schedule a meeting at 2pm"* or *"Hey calendar, move deep work to 16:00"*).
5. The voice agent turns your speech into text, then POSTs it to Pi 1.
6. Pi 1 applies schedule updates using CalDAV and calls Pi 2's `/speak` API with a summary of the changes (e.g., "I have 3 updates to your schedule").
7. Pi 2 will automatically enqueue and speak the LLM's response out loud through your connected speaker.

### Troubleshooting Audio on Pi 2
* **Microphone not detected**: Run `arecord -l` to ensure your USB mic is registered by the OS.
* **No Sound**: Run `aplay -l` to check playback devices. You might need to configure `raspi-config` to force audio out the 3.5mm jack or HDMI depending on your speaker.
* **TTS Initialization Error**: If `pyttsx3` fails to load the engine on your specific Pi OS, the script will automatically fallback to routing the string directly to the `espeak` CLI utility.
