import RAG
import SpeechRecognition
import TTS
import os
import threading
import time
import pyaudio
import keyboard


class Persona:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.stop_event = threading.Event()
        

    def start_conversation(self):
        rag = RAG.RAG(api_key=self.api_key, modelName="gpt-4")
        conversation_chain = rag.createConversationChain()
        speech_rec = SpeechRecognition.SpeechRecognition(api_key=self.api_key, model="whisper-1")
        tts = TTS.TTS()

        p = pyaudio.PyAudio()


        # format=pyaudio.paInt16: The format of the audio. paInt16 means each audio sample is represented by 16-bit integers.
        # frames_per_buffer=1024: This specifies the number of frames per buffer, which controls how much audio data is processed at a time. higher FPB --> less IO writes & higher latency
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

        input_thread = threading.Thread(target=speech_rec.record_audio, args=(p, stream, self.stop_event))
        processing_thread = threading.Thread(target=speech_rec.process_input, args=(rag, tts, conversation_chain, self.stop_event))

        # Start the threads
        print("Conversation Begins")
        input_thread.start()
        processing_thread.start()

        # Use keyboard listener for exit functionality
        while True:
            if keyboard.is_pressed('esc'):
                self.stop_event.set()  # Stop the threads
                break
            time.sleep(0.1)

        # Join the threads (optional, to keep them running)
        input_thread.join()
        processing_thread.join()

        print("Conversation terminated.")
        stream.stop_stream()
        stream.close()

        p.terminate()
        

if __name__ == "__main__":
    key = input("Please enter your OpenAI API key: ")
    persona = Persona(key)
    persona.start_conversation()