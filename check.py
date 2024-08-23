import pyaudio

if __name__ == '__main__':

    p = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device ID {i}: {info['name']} - {info['maxInputChannels']} channels")
