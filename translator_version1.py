import tkinter as tk
from tkinter import ttk
from transformers import MarianMTModel, MarianTokenizer
import torch
import speech_recognition as sr
import threading

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TranslatorApp:
    if torch.cuda.is_available():
        print("Torch is available with CUDA")

    def __init__(self, master):
        self.device = device
        self.model = None
        self.tokenizer = None

        self.master = master
        self.master.title("Neural Machine Translation System")
        self.master.geometry("1000x700")

        # Load models once at startup
        self.language_pairs = {
            'English to Spanish': 'Helsinki-NLP/opus-mt-en-es',
            'Spanish to English': 'Helsinki-NLP/opus-mt-es-en',
            'English to French': 'Helsinki-NLP/opus-mt-en-fr',
            'French to English': 'Helsinki-NLP/opus-mt-fr-en',
            'English to German': 'Helsinki-NLP/opus-mt-en-de',
            'German to English': 'Helsinki-NLP/opus-mt-de-en',
            'English to Chinese': 'Helsinki-NLP/opus-mt-en-zh',
            'Chinese to English': 'Helsinki-NLP/opus-mt-zh-en',
            'English to Arabic': 'Helsinki-NLP/opus-mt-en-ar',
            'Arabic to English': 'Helsinki-NLP/opus-mt-ar-en',
            'English to Italian': 'Helsinki-NLP/opus-mt-en-it',
            'Italian to English': 'Helsinki-NLP/opus-mt-it-en',
            'English to Portuguese': 'Helsinki-NLP/opus-mt-tc-big-en-pt',
            'Portuguese to English': 'Helsinki-NLP/opus-mt-tc-big-en-pt',
            'English to Russian': 'Helsinki-NLP/opus-mt-en-ru',
            'Russian to English': 'Helsinki-NLP/opus-mt-ru-en',
            'English to Japanese': 'Helsinki-NLP/opus-mt-en-jap',
            'Japanese to English': 'Helsinki-NLP/opus-mt-jap-en',
            'English to Hindi': 'Helsinki-NLP/opus-mt-en-hi',
            'Hindi to English': 'Helsinki-NLP/opus-mt-hi-en',
            'English to Turkish': 'Helsinki-NLP/opus-mt-en-trk',
            'Turkish to English': 'Helsinki-NLP/opus-mt-tr-en',
            'Bengali to English': 'Helsinki-NLP/opus-mt-bn-en'
        }

        # Create UI components
        self.create_widgets()

    def create_widgets(self):
        # Input text variable
        self.input_text = tk.StringVar()

        # Language selection
        tk.Label(self.master, text="Select Language Pair").pack(pady=10)
        self.lang_pair = ttk.Combobox(self.master, values=list(self.language_pairs.keys()))
        self.lang_pair.set("English to French")  # Default language pair
        self.lang_pair.pack()

        # Input box for text
        tk.Label(self.master, text="Input Text").pack(pady=5)
        input_entry = tk.Entry(self.master, textvariable=self.input_text, width=50)
        input_entry.pack()

        # Button for voice input
        voice_btn = tk.Button(self.master, text="Voice Input", command=self.recognize_speech)
        voice_btn.pack(pady=10)

        # Button for translation
        translate_btn = tk.Button(self.master, text="Translate", command=self.translate_text)
        translate_btn.pack(pady=10)

        # Output box for translated text (use a Text widget)
        tk.Label(self.master, text="Translated Text").pack(pady=5)
        self.output_textbox = tk.Text(self.master, width=100, height=20, bg="lightgray", wrap=tk.WORD)
        self.output_textbox.pack()

    # Function to handle translation in a separate thread
    def translate_text(self):
        thread = threading.Thread(target=self.perform_translation)
        thread.start()

    # Actual translation logic
    def perform_translation(self):
        # Check if the model is loaded
        if self.model is None:
            source_to_target = self.lang_pair.get()
            self.model_name = self.language_pairs[source_to_target]

            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name).to(self.device)

        # Get the input text
        text = self.input_text.get()

        # Tokenize and move input to GPU
        tokenized_text = self.tokenizer([text], return_tensors='pt').to(self.device)

        # Generate translation
        translation = self.model.generate(**tokenized_text)

        # Decode the output
        translated_text = self.tokenizer.decode(translation[0], skip_special_tokens=True)

        # Update the output text widget on the main thread
        self.master.after(0, self.update_output_text, translated_text)

    # Function to update the output text in the Text widget
    def update_output_text(self, text):
        self.output_textbox.delete(1.0, tk.END)  # Clear previous text
        self.output_textbox.insert(tk.END, text)  # Insert new text
        self.output_textbox.yview(tk.END)  # Scroll to the end if the text is too long

    # Initialize Speech Recognition
    def recognize_speech(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.output_textbox.delete(1.0, tk.END)
            self.output_textbox.insert(tk.END, "Listening...\n")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                self.input_text.set(text)
            except sr.UnknownValueError:
                self.input_text.set("Sorry, I didn't catch that.")
            except sr.RequestError:
                self.input_text.set("API unavailable.")

# Create the main window
root = tk.Tk()
app = TranslatorApp(root)

# Run the application
root.mainloop()
