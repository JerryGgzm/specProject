from openai import OpenAI
import pyaudio
import keyboard as kb
from pynput import keyboard

class TTS:
    def __init__(self):
        self.client = OpenAI()
        self.p = pyaudio.PyAudio()

    def convert(self, input_text=''):
        listening = False

        def on_press(key):
            nonlocal listening
            try:
                if key.char == 's':  # Start listening when 's' is pressed
                    if not listening:
                        print("Listening...")
                        listening = True
            except AttributeError:
                pass

        listener = keyboard.Listener(on_press=on_press)
        listener.start()

        stream = self.p.open(format=pyaudio.paInt16,  # Updated format to 16-bit integer
                             channels=1,
                             rate=24000,
                             output=True)

        with self.client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice="alloy",
                input=input_text,
                response_format="pcm"
        ) as response:
            for chunk in response.iter_bytes(1024):
                # if the user start to speak or the user want to terminate the conversation, stop output audio
                if listening or kb.is_pressed('esc'):
                    break
                stream.write(chunk)

        stream.stop_stream()
        stream.close()

        self.p.terminate()


