# Real-Time Norwegian Speech Transcription and Translation

This project captures real-time audio, transcribes Norwegian speech into text, and then translates that text into English. The system leverages state-of-the-art machine learning models for automatic speech recognition (ASR) and translation.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Overview

This project captures audio input from a microphone, transcribes it into Norwegian text using a Wav2Vec2 model, and then translates it into English using an M2M100 model. The process is done in real-time, making it suitable for live applications.

### Workflow

1. **Audio Capture:** The system captures audio from the default input device (microphone).
2. **Speech Detection:** A basic energy threshold is used to detect speech versus silence.
3. **Transcription:** Norwegian speech is transcribed to text using a pretrained Wav2Vec2 model.
4. **Translation:** The transcribed text is translated from Norwegian to English using a pretrained M2M100 model.
5. **Output:** The transcription and translation are printed to the console.

## Requirements

- Python 3.8+
- `torch` (PyTorch)
- `transformers` (Hugging Face Transformers)
- `pyaudio`
- `numpy`
- CUDA-enabled GPU (optional for faster processing)

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/norwegian-speech-translator.git
    cd norwegian-speech-translator
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Install PyAudio:**

    - For Windows:

      ```bash
      pip install pipwin
      pipwin install pyaudio
      ```

    - For macOS:

      ```bash
      brew install portaudio
      pip install pyaudio
      ```

    - For Linux:

      ```bash
      sudo apt-get install portaudio19-dev
      pip install pyaudio
      ```

## Usage

1. **Run the script:**

    ```bash
    python main.py
    ```

2. **Start speaking in Norwegian.** The system will transcribe and translate your speech in real-time.

3. **To stop the script,** press `Ctrl+C`.

## Troubleshooting

- **Error loading models:** Ensure that you have a stable internet connection when running the script for the first time, as the models need to be downloaded.
- **Device index error:** If the script cannot find the correct audio input device, you might need to adjust the `input_device_index` in the script to match your system's configuration.
- **Slow performance:** If you experience lag or slow performance, consider using a machine with a CUDA-enabled GPU for faster processing.

## Acknowledgements

- [Hugging Face](https://huggingface.co) for providing the `transformers` library and pretrained models.
- [PyTorch](https://pytorch.org) for the deep learning framework.
- [PortAudio](http://www.portaudio.com/) and [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) for audio capturing.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
