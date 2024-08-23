import wave

import pyaudio
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, M2M100ForConditionalGeneration, M2M100Tokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained models and processors
try:
    wav2vec_processor = Wav2Vec2Processor.from_pretrained("NbAiLab/nb-wav2vec2-1b-bokmaal")
    wav2vec_model = Wav2Vec2ForCTC.from_pretrained("NbAiLab/nb-wav2vec2-1b-bokmaal").to(device)
    translation_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    translation_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please check your internet connection and make sure you have the latest version of the transformers library.")
    exit(1)
# Set up PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=3)

def is_speech(frame, energy_threshold=300):
    """Detects if a given frame contains speech based on energy."""
    frame_data = np.frombuffer(frame, dtype=np.int16)
    energy = np.sum(np.abs(frame_data)) / len(frame_data)
    return energy > energy_threshold

def transcribe_audio(speech_buffer):
    """Transcribe Norwegian audio to text."""
    speech = np.frombuffer(b''.join(speech_buffer), dtype=np.int16)
    input_values = wav2vec_processor(speech, return_tensors="pt", sampling_rate=16000).input_values

    # Ensure the input values are in float32 format
    input_values = input_values.to(device).float()

    with torch.no_grad():
        logits = wav2vec_model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = wav2vec_processor.decode(predicted_ids[0])
    return transcription


def translate_text(text):
    """Translate Norwegian text to English."""
    # Set the source and target language
    translation_tokenizer.src_lang = "no"  # Norwegian
    encoded_no = translation_tokenizer(text, return_tensors="pt").to(device)

    # Generate the translation
    generated_tokens = translation_model.generate(
        **encoded_no,
        forced_bos_token_id=translation_tokenizer.get_lang_id("en")  # English
    )

    # Decode the generated tokens to English text
    translated_text = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

    return translated_text



def capture_and_translate():
    """Continuously capture audio, transcribe it, and translate it."""
    speech_buffer = []
    silence_buffer = []
    silence_threshold = 30  # Number of silent frames to consider as end of speech

    print("Listening... (Press Ctrl+C to stop)")

    while True:
        frame = stream.read(1024)
        if is_speech(frame):
            speech_buffer.append(frame)
            silence_buffer.clear()  # Reset silence buffer when speech is detected
        else:
            silence_buffer.append(frame)

            # If enough silence is detected, consider the speech as ended
            if len(silence_buffer) > silence_threshold and speech_buffer:
                # Process the captured speech
                transcription = transcribe_audio(speech_buffer)
                print("Transcription (Norwegian):", transcription)

                translated_text = translate_text(transcription)
                print("Translated Text (English):", translated_text)

                # Reset buffers for the next speech segment
                speech_buffer.clear()
                silence_buffer.clear()

if __name__ == '__main__':
    try:
        capture_and_translate()
    except KeyboardInterrupt:
        print("\nTerminating...")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()



