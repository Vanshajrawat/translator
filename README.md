# 🌐 Neural Machine Translation Desktop App

A multilingual desktop application that translates text and voice input across 30+ language pairs using Hugging Face’s MarianMT transformer models. Built with PyQt5, this app provides real-time translation, voice recognition, and voice output in a modern GUI interface.

## 🚀 Features

- 🌍 **Supports 30+ Language Pairs** (e.g., English ⇌ Spanish, Hindi, German, French, Arabic, Chinese, etc.)
- 🧠 **Powered by Transformers**: Uses MarianMT models from Hugging Face for accurate translations.
- 🎤 **Voice Input**: Speak to translate using the microphone (SpeechRecognition).
- 🔊 **Voice Output**: Translated text is spoken back using `pyttsx3` (Text-to-Speech).
- 🎨 **Animated UI**: Smooth visual transitions and responsive design using PyQt5.
- ⚡ **Multithreaded Translation**: UI remains responsive during translation via background threading.
- 💻 **Offline Execution**: No web server required; runs fully on your machine.

## 🛠️ Tech Stack

- **Language**: Python 3
- **GUI**: PyQt5
- **Translation Models**: Hugging Face Transformers (`MarianMTModel`)
- **Speech Recognition**: `speech_recognition`
- **Text-to-Speech**: `pyttsx3`
- **Device Support**: CPU and GPU (via PyTorch)

## 📦 Installation

1. **Clone the repository:**
   
   git clone https://github.com/yourusername/translator-app.git
   cd translator-app
