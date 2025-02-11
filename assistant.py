import base64
from threading import Lock, Thread
from faster_whisper import WhisperModel
import numpy as np
from PIL.Image import Image
# Load environment variables from a .env file
from dotenv import load_dotenv

# Import OpenCV for video capture and image processing
import cv2
from cv2 import VideoCapture, imencode

# Import OpenAI library for text-to-speech functionality
import openai

# Import LangChain components for creating chat chains and message histories
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

# Import LangChain OpenAI integration for chat models
from langchain_openai import ChatOpenAI

# Import PyAudio and speech_recognition libraries for audio processing
from pyaudio import PyAudio, paInt16
import speech_recognition as sr
from faster_whisper import WhisperModel
import io

# Constants
AUDIO_FORMAT = paInt16
CHANNELS = 1
RATE = 23000
CHUNK_SIZE = 1024
TTS_MODEL = "tts-1"
TTS_VOICE = "nova"
TTS_RESPONSE_FORMAT = "pcm"

# Load environment variables from the .env file
load_dotenv()


# Class to handle webcam video stream
class ScreenStream:
    def __init__(self):
        from threading import Lock
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
        from PIL import Image

        with mss.mss() as sct:  # Create the mss instance inside the thread
            monitor = sct.monitors[1]
            while self.running:
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                with self.lock:
                    self.frame = frame  # Ensure frame is updated

    def read(self, encode=False):
        with self.lock:
            if self.frame is None:  # Prevent AttributeError
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
    def __init__(self, model):
        self.chain = self._create_inference_chain(model)

    def answer(self, prompt, image):
        if not prompt:
            return

        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)

        if response:
            self._tts(response)

    def _tts(self, response):
        player = PyAudio().open(format=AUDIO_FORMAT, channels=CHANNELS, rate=RATE, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            response_format=TTS_RESPONSE_FORMAT,
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, model):
        SYSTEM_PROMPT = """
        You are a witty assistant that will use the chat history and the image 
        provided by the user to answer its questions. Your job is to answer 
        questions.

        Use few words on your answers. Go straight to the point. Do not use any
        emoticons or emojis. 

        Be friendly and helpful. Show some personality. Be cooperative.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | model | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


screen_stream = ScreenStream().start()

#model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# You can use OpenAI's GPT-4o model instead of Gemini Flash
# by uncommenting the following line:
model = ChatOpenAI(model="gpt-4o")

assistant = Assistant(model)



def audio_callback(recognizer, audio):
    try:
        # Ensure `audio` is a valid AudioData object
        if not isinstance(audio, sr.AudioData):
            print("❌ Invalid audio object received.")
            return

        # Convert to file-like object for Faster-Whisper
        wav_data = io.BytesIO(audio.get_wav_data(convert_rate=16000, convert_width=2))

        # Load the Faster-Whisper model
        model = WhisperModel("large-v2")

        # Transcribe with a specific language (e.g., "pl" for Polish)
        segments, _ = model.transcribe(wav_data, language="pl")

        # Convert the transcribed segments into text
        prompt = " ".join(segment.text for segment in segments)
        print("📝 Recognized:", prompt)

        # Send the transcribed text to the assistant
        assistant.answer(prompt, screen_stream.read(encode=True))

    except Exception as e:
        print(f"❌ Error in recognition: {e}")


recognizer = sr.Recognizer()
mic = sr.Microphone()

with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2)
    print("🎙️ Say something...")
    audio = recognizer.listen(source)
    print("✅ Captured audio!")

    try:
        # Convert to text using Google's default STT (for testing)
        text = recognizer.recognize_google(audio, language="pl-PL")
        print("📝 Recognized text:", text)
    except sr.UnknownValueError:
        print("❌ Could not understand audio.")
    except sr.RequestError as e:
        print(f"❌ API error: {e}")

stop_listening = recognizer.listen_in_background(mic, audio_callback)

while True:
    frame = screen_stream.read()
    if frame is not None:
        cv2.imshow("screen", frame)

    if cv2.waitKey(1) in [27, ord("q")]:
        break

screen_stream.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
