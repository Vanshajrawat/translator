import sys
from PyQt5.QtWidgets import (QApplication, QWidget, 
                             QLabel, QComboBox, 
                             QVBoxLayout, QLineEdit, 
                             QPushButton, QTextEdit, 
                             QGraphicsOpacityEffect,
                             QHBoxLayout, QSizePolicy)
from PyQt5.QtCore import (QThread, pyqtSignal, 
                          QPropertyAnimation, QEasingCurve)
from PyQt5.QtGui import (QPixmap, QFont, QIcon, 
                         QPalette, QBrush, QColor)
from PyQt5.QtCore import (Qt, QPropertyAnimation, 
                          pyqtSlot, QEasingCurve)
from transformers import MarianMTModel, MarianTokenizer
import torch
import speech_recognition as sr
import pyttsx3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TranslatorApp(QWidget):
    def __init__(self):
        super().__init__()

        self.device = device
        self.model = None
        self.tokenizer = None

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
            'Bengali to English': 'Helsinki-NLP/opus-mt-bn-en',
            'English to Vietnamese': 'Helsinki-NLP/opus-mt-en-vi',
            'English to Indonesian': 'Helsinki-NLP/opus-mt-en-id',
            'English to Czech': 'Helsinki-NLP/opus-mt-en-cs',
            'English to Romanian': 'Helsinki-NLP/opus-mt-en-ro',
            'English to Swedish': 'Helsinki-NLP/opus-mt-en-sv',
            'English to Catalan': 'Helsinki-NLP/opus-mt-en-ca',
            'English to Finnish': 'Helsinki-NLP/opus-mt-en-fi',
            'English to Danish': 'Helsinki-NLP/opus-mt-en-da',
            'English to Ukrainian': 'Helsinki-NLP/opus-mt-en-uk',
            'English to Hungarian': 'Helsinki-NLP/opus-mt-en-hu',
            'English to Greek': 'Helsinki-NLP/opus-mt-en-el',
            'English to Urdu': 'Helsinki-NLP/opus-mt-en-ur',
            'English to Hebrew': 'Helsinki-NLP/opus-mt-en-he',
            'English to Bulgarian': 'Helsinki-NLP/opus-mt-en-he'
        }

        # Set up the UI with animations and visuals
        self.initUI()


    def initUI(self):
        self.setWindowTitle('Neural Machine Translation System')
        self.setGeometry(100, 100, 1000, 700)
        self.setWindowIcon(QIcon("images/icon.jpg"))

        # Create the main layout (Vertical Layout)
        layout = QVBoxLayout()  

        # Create a Horizontal Layout (hbox) to hold the label and combobox side by side
        hbox = QHBoxLayout()

        # Language selection label
        self.lang_label = QLabel('Select Language Pair')
        self.lang_label.setFont(QFont('Arial', 20))
        self.lang_label.setStyleSheet(
            "font-size: 20px;"
            "font-family: fantasy;"
            "color: black;"
            "background-color: hsl(217, 83%, 77%);"
            "font-weight: bold;"
            "font-style: italic;"
            "padding: 15px 50px;"  # Adjusted padding for smaller width
            "margin: 25px;"
            "border: 2px solid;"
            "border-radius: 8px;"
        )

        self.lang_label.adjustSize()
        self.lang_label.setAlignment(Qt.AlignLeft)

        # Add the label to the horizontal layout (hbox)
        hbox.addWidget(self.lang_label, 1)  # Stretch factor 1 (smaller)

        # Language selection combo box
        self.lang_pair = QComboBox()
        self.lang_pair.addItems(self.language_pairs)  # Add language pairs here
        self.lang_pair.setCurrentText("English to Spanish")  # Default language pair
        self.lang_pair.setStyleSheet(
            "font-size: 20px;"
            "font-family: fantasy;"
            "color: #333;"
            "background-color: hsl(217, 83%, 77%);"
            "font-weight: bold;"
            "padding: 15px 75px;"
            "margin: 25px;"
            "border: 2px solid;"
            "border-radius: 8px;"
        )
        

        # Add the combo box to the horizontal layout (hbox)
        hbox.addWidget(self.lang_pair, 2)  # Stretch factor 2 (larger)

        # Add the horizontal layout (hbox) to the main layout (layout)
        layout.addLayout(hbox)

        # Set the layout for the main window
        self.setLayout(layout)

        # Input text horizontal layout to stack the label and text box side by side
        input_layout = QHBoxLayout()

        # Input text label
        self.input_label = QLabel('Input Text')
        self.input_label.setStyleSheet(
            "font-size: 20px;"
            "font-family: fantasy;"
            "background-color: hsl(217, 93%, 71%);"
            "font-weight: bold;"
            "font-style: italic;"
            "padding: 15px 30px;"  # Adjust padding for smaller left text
            "margin: 25px;"
            "border: 3px solid;"
            "border-radius: 8px;"
        )

        self.input_label.adjustSize()
        self.input_label.setAlignment(Qt.AlignLeft)
        self.input_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)  # Set smaller size for label
        input_layout.addWidget(self.input_label)

        # Input text box (QLineEdit)
        self.input_text = QLineEdit()
        self.input_text.setStyleSheet(
            "font-size: 20px;"
            "font-family: fantasy;"
            "background-color: hsl(217, 93%, 71%);"
            "font-weight: bold;"
            "padding: 15px 75px;"
            "margin: 25px;"
            "border: 3px solid;"
            "border-radius: 8px;"
        )
        
        self.input_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)  # Make the input field larger
        input_layout.addWidget(self.input_text)

        # Add the horizontal layout to the main layout
        layout.addLayout(input_layout)

        # Set the main layout to the window
        self.setLayout(layout)

        # Create a horizontal layout to hold both buttons side by side
        button_hbox = QHBoxLayout()

        # Voice input button
        self.voice_btn = QPushButton('Voice Input')

        # Setting name for using CSS properties
        self.voice_btn.setObjectName("voice_button")
        self.voice_btn.clicked.connect(self.recognize_speech)

        # Add voice button to the horizontal layout
        button_hbox.addWidget(self.voice_btn)

        # Translate button
        self.translate_btn = QPushButton('Translate')

        # Setting name for using CSS properties
        self.translate_btn.setObjectName("translate_btn")
        self.translate_btn.clicked.connect(self.translate_text)

        # Add translate button to the horizontal layout
        button_hbox.addWidget(self.translate_btn)

        # Add the horizontal button layout to the main vertical layout
        layout.addLayout(button_hbox)

        # Using CSS properties on buttons
        self.setStyleSheet("""
            QPushButton{
                font-size: 20px;
                font-family: fantasy;
                padding: 15px 75px;
                margin: 25px;
                border: 3px solid;
                border-radius: 30px;
                font-weight: bold;
                font-style: italic;
            }

            QPushButton#voice_button{
                background-color: hsl(217, 95%, 62%);
            }

            QPushButton#translate_btn{
                background-color: hsl(217, 88%, 58%);
            }

            QPushButton#voice_button:hover{
                background-color: hsl(217, 95%, 44%);
            }

            QPushButton#translate_btn:hover{
                background-color: hsl(217, 88%, 40%);
            }
        """)

        self.voice_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.translate_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


        # Output text label
        self.output_label = QLabel('Translated Text')
        self.output_label.setStyleSheet(
            "font-size: 20px;"
            "font-family: fantasy;"
            "font-style: italic;"
            "background-color: hsl(217, 91%, 54%);"
            "font-weight: bold;"
            "padding: 15px 75px;"
            "margin: 25px;"
            "border: 3px solid;"
            "border-radius: 8px;"
        )

        # Add the output text box to the layout
        output_hbox = QHBoxLayout()  # Horizontal layout for stacking side by side

        # Adjust the size of the label to fit the content
        self.output_label.adjustSize()  # Ensures the label occupies only the necessary space
        output_hbox.addWidget(self.output_label, 1)  # Set stretch factor to 1 (smaller width)

        # Output text box initialization in UI setup
        self.output_textbox = QTextEdit()
        self.output_textbox.setStyleSheet(
            "font-size: 15px;"
            "font-family: fantasy;"
            "background-color: hsl(217, 91%, 54%);"
            "font-weight: bold;"
            "padding: 15px 75px;"
            "margin: 25px;"
            "border: 3px solid;"
            "border-radius: 8px;"
        )

        self.output_textbox.setReadOnly(True)  # Make it read-only

        output_hbox.addWidget(self.output_textbox, 5)  # Set stretch factor to 3

        # Add the horizontal layout to the main layout
        layout.addLayout(output_hbox)

        # Set the layout to the window
        self.setLayout(layout)


    def update_background_image(self):
        # Load the background image
        background_image = QPixmap('images/background_image2.jpg')

        # Resize the background image to match the widget's current size
        scaled_image = background_image.scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)

        # Set the background image using QPalette
        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(scaled_image))
        self.setPalette(palette)

        # background_image.lower()  # Make sure the background is behind other widgets

        # Add a background image
        # background_label = QLabel(self)
        # pixmap = QPixmap("background_image1.jpg")  # Replace with your image path
        # background_label.setPixmap(pixmap)
        # background_label.resize(self.width(), self.height())
        # background_label.setScaledContents(True)

    def resizeEvent(self, event):
        # Reapply the background image when the window is resized
        self.update_background_image()
        return super().resizeEvent(event)
        

    def animate_output(self):
        # Animate the opacity of the output textbox
        effect = QGraphicsOpacityEffect()
        self.output_textbox.setGraphicsEffect(effect)
        self.animation = QPropertyAnimation(effect, b"opacity")
        self.animation.setDuration(1000)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.start()


    def translate_text(self):
        input_text = self.input_text.text()
        lang_pair = self.lang_pair.currentText()

        # Start translation in a separate thread
        self.translation_thread = TranslationThread(input_text, lang_pair, self.language_pairs, self.device)
        self.translation_thread.translation_done.connect(self.update_output_text)
        self.translation_thread.start()

        # Start loading animation (fade out the text box to indicate it's working)
        self.animate_output()


    # Function to provide text and voice output
    def provide_output(self, translated_text):
        # Text output
        self.output_textbox.clear()
        # Output translated text to the text box
        self.output_textbox.setPlainText(translated_text)
        
        # Voice output
        # Setting up text-to-speech engine
        self.tts_engine = pyttsx3.init()

        # Slowing down the speech rate (default is around 200 words per minute)
        self.tts_engine.setProperty('rate', 125)  # Lower the value for slower speech, e.g., 150

        self.tts_engine.setProperty('volume', 0.9)  # Set volume between 0.0 and 1.0
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[1].id)  # Choose a specific voice (male/female)

        self.tts_engine.say(translated_text)
        self.tts_engine.runAndWait()

    def update_output_text(self, translated_text):
        self.provide_output(translated_text)
        self.animate_output()  # Animate when the text appears


    def recognize_speech(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.output_textbox.setPlainText("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                self.input_text.setText(text)
            except sr.UnknownValueError:
                self.input_text.setText("Sorry, I didn't catch that.")
            except sr.RequestError:
                self.input_text.setText("API unavailable.")


    # Create animation for the voice input button (color animation)
    def animate_voice_input(self):
        animation = QPropertyAnimation(self.voice_btn, b"backgroundColor")
        animation.setDuration(1000)  # Animation duration in milliseconds
        animation.setStartValue(QColor(255, 165, 0))  # Starting color (Orange)
        animation.setEndValue(QColor(34, 139, 34))    # Ending color (Green)
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.setLoopCount(3)  # Loop the animation 3 times
        animation.start()

        # You can also connect this function to your voice button when clicked
        self.voice_btn.clicked.connect(self.animate_voice_input)


    # Create animation for the voice output button (opacity animation)
    def animate_voice_output(self):
        # Apply opacity effect to the button
        opacity_effect = QGraphicsOpacityEffect()
        self.translate_btn.setGraphicsEffect(opacity_effect)

        # Create an opacity animation
        opacity_animation = QPropertyAnimation(opacity_effect, b"opacity")
        opacity_animation.setDuration(1000)  # Animation duration in milliseconds
        opacity_animation.setStartValue(0.1)  # Start with 10% opacity (fade-in effect)
        opacity_animation.setEndValue(1.0)  # End with 100% opacity
        opacity_animation.setEasingCurve(QEasingCurve.InOutQuad)
        opacity_animation.setLoopCount(3)  # Loop the animation 3 times
        opacity_animation.start()

        # Connect this function to your voice button when clicked
        self.translate_btn.clicked.connect(self.animate_voice_output)


    def animate_voice_io(self):
    # Animate voice input button (color)
        self.animate_voice_input()
        
        # Animate voice output button (opacity)
        self.animate_voice_output()

        # You can connect this to both buttons or when either is clicked
        self.voice_btn.clicked.connect(self.animate_voice_io)
        self.translate_btn.clicked.connect(self.animate_voice_io)



class TranslationThread(QThread):
    translation_done = pyqtSignal(str)


    def __init__(self, text, lang_pair, language_pairs, device):
        super().__init__()
        self.text = text
        self.lang_pair = lang_pair
        self.language_pairs = language_pairs
        self.device = device


    def run(self):
        model_name = self.language_pairs[self.lang_pair]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(self.device)

        tokenized_text = tokenizer([self.text], return_tensors='pt').to(self.device)
        translation = model.generate(**tokenized_text)

        translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
        self.translation_done.emit(translated_text)


# Running the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    translator = TranslatorApp()
    translator.show()
    sys.exit(app.exec_())
