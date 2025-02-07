import base64
import numpy as np
import cv2
import ollama
import pyttsx3
from threading import Lock, Thread
from PIL import Image
from cv2 import imencode
from speech_recognition import Microphone, Recognizer, UnknownValueError, RequestError
import pyaudio

# Constants for audio processing
AUDIO_FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio
RATE = 48000  # Increased to 48 kHz for better recognition
CHUNK_SIZE = 2048  # Larger chunk size for smoother recognition

# Initialize text-to-speech engine
engine = pyttsx3.init()

# ==========================
# SCREEN CAPTURING CLASS
# ==========================
class ScreenStream:
    """ Continuously captures the screen for AI analysis. """

    def __init__(self):
        self.frame = None
        self.running = False
        self.lock = Lock()

    def start(self):
        """ Starts the screen capture thread. """
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        """ Captures the screen in real-time. """
        import mss
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary screen
            while self.running:
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                with self.lock:
                    self.frame = frame

    def read(self, encode=False):
        """ Returns the latest frame, optionally encoded as Base64. """
        with self.lock:
            if self.frame is None:
                return None
            frame = self.frame.copy()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        """ Stops the screen capture. """
        self.running = False
        if self.thread.is_alive():
            self.thread.join()

# ==========================
# AI ASSISTANT CLASS
# ==========================
class Assistant:
    """ AI Assistant that processes speech, screen capture, and generates responses. """

    def __init__(self, model_name="llava:34b"):
        self.model_name = model_name

    def answer(self, prompt, image):
        """ Generates a response based on user query and screen capture. """
        if not prompt:
            return

        print("Prompt:", prompt)

        messages = [
            {"role": "system", "content": "You are an AI assistant capable of analyzing images and text."},
            {"role": "user", "content": prompt, "images": [image.decode()] if image else []}
        ]

        response = ollama.chat(
            model=self.model_name,
            messages=messages
        )["message"]["content"]

        print("Response:", response)
        self._tts(response)

    def _tts(self, response):
        """ Converts text to speech for responses. """
        engine.say(response)
        engine.runAndWait()

# ==========================
# SPEECH RECOGNITION OPTIMIZATION
# ==========================
def audio_callback(recognizer, audio):
    """ Converts speech to text and sends it to AI assistant. """
    try:
        # Use Whisper with a more accurate model
        prompt = recognizer.recognize_whisper(audio, model="medium", language="polish")

        # Capture screen and send both text & image to AI
        assistant.answer(prompt, screen_stream.read(encode=True))

    except UnknownValueError:
        print("Sorry, could not understand the audio.")
    except RequestError as e:
        print(f"Speech recognition error: {e}")

# ==========================
# MAIN PROGRAM
# ==========================
screen_stream = ScreenStream().start()
assistant = Assistant(model_name="llava:34b")

# Initialize speech recognizer with optimized settings
recognizer = Recognizer()
microphone = Microphone()

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)  # Auto-adjust for background noise
    recognizer.energy_threshold = 300  # Set threshold for voice detection
    recognizer.dynamic_energy_threshold = True  # Enable dynamic thresholding

# Start background listening
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# ==========================
# MAIN LOOP (SHOW SCREEN CAPTURE)
# ==========================
while True:
    frame = screen_stream.read()
    if frame is not None:
        cv2.imshow("screen", frame)

    if cv2.waitKey(1) in [27, ord("q")]:  # Exit on 'ESC' or 'Q'
        break

# Cleanup
screen_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
