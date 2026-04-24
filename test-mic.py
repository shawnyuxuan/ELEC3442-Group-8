from speech_to_text import VoiceAssistant

assistant = VoiceAssistant(
    model_path = "/home/tamim/Desktop/ELEC3442-Group-8/vosk-model-small-en-us-0.15",
    enable_tts = False
    )

text = assistant.listen(seconds=3)
print("Final text:", text)