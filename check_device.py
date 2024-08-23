import wave

import pyaudio


def record_audio(device_id, duration=5, filename="test_output.wav"):
    """Record audio from the specified device ID for a given duration."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024,
                    input_device_index=device_id)

    frames = []
    print("Recording...")

    for _ in range(0, int(16000 / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording finished.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()


def replay_audio(filename="test_output.wav"):
    """Replay the recorded audio from the WAV file."""
    p = pyaudio.PyAudio()
    wf = wave.open(filename, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    print("Replaying audio...")

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Playback finished.")


if __name__ == '__main__':
    # Replace 'your_device_id' with the correct device ID of your microphone
    device_id =12  # Set this to the correct ID
    duration = 10 # Duration in seconds for recording

    # Step 1: Record audio from the specified device
    record_audio(device_id, duration=duration)

    # Step 2: Replay the recorded audio
    replay_audio()
