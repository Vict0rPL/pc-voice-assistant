import base64
from threading import Lock, Thread

import numpy as np
from PIL import Image
import cv2
from cv2 import VideoCapture, imencode

import ollama  # Ollama for local AI inference
import pyttsx3  # Local Text-to-Speech (instead of OpenAI API)

from pyaudio import PyAudio, paInt16
from speech_recognition import Microphone, Recognizer, UnknownValueError

# Constants
AUDIO_FORMAT = paInt16
CHANNELS = 1
RATE = 24000
CHUNK_SIZE = 1024

# Local TTS Engine
engine = pyttsx3.init()

# Class to handle screen video stream
class ScreenStream:
    def __init__(self):
        self.frame = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            while self.running:
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                with self.lock:
                    self.frame = frame

    def read(self, encode=False):
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

class Assistant:
    def __init__(self, model_name="llama3"):
        self.model_name = model_name

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        # Ollama does not support image input natively (except for multimodal models)
        response = ollama.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )["message"]["content"]

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        """Local text-to-speech using pyttsx3"""
        engine.say(response)
        engine.runAndWait()

# Start screen capturing
screen_stream = ScreenStream().start()

# Use a locally installed Ollama model
assistant = Assistant(model_name="llava:34b")

def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, screen_stream.read(encode=True))
    finally:
        pass

recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)

while True:
    frame = screen_stream.read()
    if frame is not None:
        cv2.imshow("screen", frame)

    if cv2.waitKey(1) in [27, ord("q")]:
        break

screen_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
