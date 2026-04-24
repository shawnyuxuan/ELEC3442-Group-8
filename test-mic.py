from src.speech_to_text import VoiceAssistant
import os

assistant = VoiceAssistant(
    model_path = os.path.join(os.getcwd(), "vosk-model-small-en-us-0.15"),
    enable_tts = False,
    is_local = False
)

text = assistant.listen(seconds=10)
print("Final text:", text)