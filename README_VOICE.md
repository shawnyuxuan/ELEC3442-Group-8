# Voice Assistant Integration (Multi-Pi Setup)

This document explains the setup and usage of the two-Pi voice assistant system for the Smart Calendar Agent. Since the main Raspberry Pi (Pi 1) operates headless (CLI only) with sensors, a secondary Raspberry Pi (Pi 2) is used exclusively for handling voice input (Speech-to-Text) and audio output (Text-to-Speech).

## Architecture Overview

The system is split into a **Frontend (Pi 2)** and a **Backend (Pi 1)**:

1. **Pi 2 (Ears & Mouth)**: Listens to user voice queries using a connected microphone, converts the audio to text using Google Web Speech API (STT), and sends the text to Pi 1. Once a response is received, it plays it through a connected speaker (TTS).
2. **Pi 1 (The Brain)**: Runs the main `calendar-agent.py`. It receives the text query via a Flask API (`/voice` endpoint). It automatically gathers the current SenseHat sensor data, last night's extracted sleep quality, and today's calendar events. It packages this context and the user's query into an LLM prompt, asks the Qwen LLM to act as a conversational assistant, and sends the plain text response back to Pi 2.

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
Ensure you have your LLM API keys exported in your environment (e.g., `QWEN_API_KEY`).
```bash
python3 calendar-agent.py
```
*Note down the IP address of Pi 1 on your local network (e.g., `192.168.1.100`). You can find this by running `hostname -I` in a separate terminal on Pi 1.*

---

## 2. Setup Pi 2 (Frontend / Voice Client)

Pi 2 only needs the `voice_client.py` script and some audio processing libraries. It does not need the heavy calendar or ML dependencies.

### Installation

First, install the system-level audio dependencies (flac is needed for the STT engine, pyaudio for mic access, and espeak as a TTS fallback):
```bash
sudo apt-get update
sudo apt-get install flac espeak python3-pyaudio
```

Next, install the Python libraries for recording, translating to text, and making HTTP requests:
```bash
pip install SpeechRecognition pyaudio pyttsx3 requests
```

### File Transfer
Copy **only** the `voice_client.py` file from this repository onto Pi 2. 

---

## 3. Usage & Interaction

With **Pi 1** already running the `calendar-agent.py` server:

1. Log into **Pi 2**.
2. Run the voice client, pointing it to Pi 1's IP Address (replace `192.168.1.100` with the actual IP of Pi 1).

```bash
python3 voice_client.py --server http://192.168.1.100:5888/voice
```

3. The script will ask you to prepare. Press **Enter** to start recording your voice.
4. **Speak your request!** (e.g., *"What is my schedule looking like today?"* or *"How did I sleep last night based on the sensor data?"*)
5. The text is processed, sent to Pi 1, evaluated alongside your live calendar/sleep state, and the text response is sent back to Pi 2.
6. Pi 2 will speak the LLM's response out loud through your connected speaker.

### Troubleshooting Audio on Pi 2
* **Microphone not detected**: Run `arecord -l` to ensure your USB mic is registered by the OS.
* **No Sound**: Run `aplay -l` to check playback devices. You might need to configure `raspi-config` to force audio out the 3.5mm jack or HDMI depending on your speaker.
* **TTS Initialization Error**: If `pyttsx3` fails to load the engine on your specific Pi OS, the script will automatically fallback to routing the string directly to the `espeak` CLI utility.
