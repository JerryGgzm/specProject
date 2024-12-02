import wave
import os
import queue
from openai import OpenAI
import io
from pynput import keyboard

class SpeechRecognition:
    def __init__(self, api_key, model):
        self.api_key = api_key
        self.input_queue = queue.Queue()
        self.client = OpenAI()
        # Set the API key as an environment variable
        os.environ["OPENAI_API_KEY"] = self.api_key

        # Optionally, check that the environment variable was set correctly
        print("OPENAI_API_KEY has been set!")

        self.model = model
        print(f"Model selected: {model}")

    def transcribe_audio_chunk(self, audio_chunk, rate=44100):
        """Transcribes a given audio chunk using OpenAI's Whisper model."""
        # Convert audio data to a numpy array
        # audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        
        # Transcribe the audio (API expects the file to be sent in bytes or an array)
        # response = openai.Audio.transcribe(model="whisper-1", file=audio)
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # Assuming 16-bit audio
            wav_file.setframerate(rate)
            wav_file.writeframes(audio_chunk)
        
        buffer.seek(0)  # Go to the beginning of the buffer
        buffer.name = "file.wav"
        response = self.client.audio.transcriptions.create(
                                        model="whisper-1", 
                                        file=buffer, 
                                        response_format="text"
                                        )
        
        # Extract text from the response
        return response

    def record_audio(self, p, stream, stop_event, fs=44100):
        """Records audio only while 's' is pressed and stops when released."""
        audio_buffer = b''
        chunk_size = 1024
        listening = False  # Flag to indicate if listening is active

        def on_press(key):
            nonlocal listening
            try:
                if key.char == 's':  # Start listening when 's' is pressed
                    if not listening:
                        print("Listening...")
                        listening = True
            except AttributeError:
                pass

        def on_release(key):
            nonlocal listening, audio_buffer
            try:
                if key.char == 's':  # Stop listening when 's' is released
                    if listening:
                        listening = False
                        print("Stopped listening")
                        
                        # Transcribe the audio only when 's' is released
                        if len(audio_buffer) > 0:
                            transcribed_text = self.transcribe_audio_chunk(audio_buffer)
                            self.input_queue.put(transcribed_text)
                            audio_buffer = b''  # Clear buffer after transcription
            except AttributeError:
                pass

        # Start listener in a non-blocking way
        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        while not stop_event.is_set():
            if listening:
                audio_data = stream.read(chunk_size)
                audio_buffer += audio_data

            # time.sleep(0.1)  # Prevents high CPU usage
    
    def process_input(self,rag, tts, conversation_chain, stop_event):
        while not stop_event.is_set():  # Run until the stop event is set
            if not self.input_queue.empty():
                # print(f"Input queue: {list(self.input_queue.queue)}")
                user_input = self.input_queue.get()
                print(f"User: {user_input}")
                rag.runConversation(tts=tts, user_input=user_input, conversation_chain=conversation_chain)


if __name__ == '__main__':
    key = input("Please enter your OpenAI API key: ")
    speechReg = SpeechRecognition(api_key=key, model="whisper-1")
    speechReg.record_audio(filename='test.wav')
