import sys
import os
import time
import wave
import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLineEdit, QLabel, QSplitter, QMessageBox, QTextBrowser, QStyleFactory
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QTextCursor, QTextBlockFormat
from faster_whisper import WhisperModel
from llama_cpp import Llama
from config.config_Meta_Llama_3_1_8B_Instruct_Q4_K_M import config
import markdown
import tiktoken

# Costanti per la gestione dei token
MAX_CONTEXT_TOKENS = 2048        # Numero massimo di token nel contesto
MAX_RESPONSE_TOKENS = 500        # Numero massimo di token nella risposta
RESERVED_TOKENS = 500            # Token riservati per prevenire superamenti

class AudioRecorder(QThread):
    finished = pyqtSignal(str)      # Segnale emesso quando la trascrizione è finita

    def __init__(self, whisper_model):
        super().__init__()
        self.whisper_model = whisper_model
        self.is_recording = False
        self.audio = None
        self.stream = None
        self.document_chunks = []

    def initialize_audio(self):
        self.audio = pyaudio.PyAudio()  # Inizializza PyAudio
        self.stream = self.audio.open(
            format=pyaudio.paInt16,     # Formato audio a 16-bit
            channels=1,                 # Canali audio (mono)
            rate=44100,                 # Frequenza di campionamento 44.1kHz
            input=True,                 # Imposta il flusso come input
            frames_per_buffer=1024      # Dimensione del buffer
        )

    def run(self):
        self.initialize_audio()
        self.frames = []
        self.is_recording = True
        while self.is_recording:
            try:
                data = self.stream.read(1024)  # Legge dati dal microfono
                self.frames.append(data)        # Aggiunge i dati al buffer
            except OSError as e:
                print(f"Error reading from stream: {e}")
                break

        self.cleanup_audio()
        
        # Salva l'audio registrato in un file WAV
        wf = wave.open("output.wav", "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b"".join(self.frames))
        wf.close()
        
        # Trascrive l'audio usando Whisper
        segments, _ = self.whisper_model.transcribe(
            "output.wav",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        transcription = " ".join([segment.text for segment in segments])
        
        self.finished.emit(transcription)  # Emissione del segnale con la trascrizione

    def stop(self):
        self.is_recording = False  # Ferma la registrazione

    def cleanup_audio(self):
        if self.stream:
            self.stream.stop_stream()   # Ferma lo stream audio
            self.stream.close()        # Chiude lo stream
        if self.audio:
            self.audio.terminate()     # Termina PyAudio

class ChatbotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalChat powered by Meta Llama 3.1")  # Titolo della finestra
        self.setGeometry(100, 100, 1200, 800)                     # Dimensioni della finestra

        self.central_widget = QWidget()            # Widget centrale
        self.setCentralWidget(self.central_widget) # Imposta il widget centrale
        self.main_layout = QHBoxLayout(self.central_widget)  # Layout orizzontale principale

        self.splitter = QSplitter(Qt.Horizontal)  # Splitter per dividere la finestra
        self.main_layout.addWidget(self.splitter) # Aggiunge lo splitter al layout principale

        self.chat_widget = QWidget()               # Widget per la chat
        self.chat_layout = QVBoxLayout(self.chat_widget)  # Layout verticale per la chat
        self.splitter.addWidget(self.chat_widget)  # Aggiunge il widget della chat allo splitter

        self.sidebar = QWidget()                   # Widget per la sidebar
        self.sidebar_layout = QVBoxLayout(self.sidebar)  # Layout verticale per la sidebar
        self.splitter.addWidget(self.sidebar)      # Aggiunge la sidebar allo splitter

        self.chat_history = []                     # Lista per memorizzare la cronologia della chat
        self.prompts = {'default': ''}            # Dizionario per i prompt (solo default)
        self.current_prompt = 'default'            # Prompt corrente

        self.settings = QSettings("YourCompany", "LocalChat")  # Impostazioni salvate
        self.last_directory = self.settings.value("last_directory", "")  # Ultima directory usata

        self.setup_chat_area()         # Configura l'area della chat
        self.setup_input_area()        # Configura l'area di input
        self.setup_sidebar()           # Configura la sidebar

        self.load_llm()                # Carica il modello Llama
        self.load_whisper()            # Carica il modello Whisper
        self.load_prompts()            # Carica i prompt

        self.audio_recorder = AudioRecorder(self.whisper_model)  # Inizializza il registratore audio
        self.audio_recorder.finished.connect(self.on_transcription_finished)  # Connessione del segnale

        self.setStyle(QStyleFactory.create('Fusion'))  # Imposta lo stile dell'applicazione

        # Altri stili (commentati)
        # import qdarkstyle
        # self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # from qt_material import apply_stylesheet
        # apply_stylesheet(self, theme='dark_teal.xml')

        # import qdarktheme
        # self.setStyleSheet(qdarktheme.load_stylesheet())


    def setup_chat_area(self):
        self.chat_area = QTextBrowser()         # Widget per visualizzare la chat
        self.chat_area.setReadOnly(True)        # Imposta l'area della chat come sola lettura
        self.chat_layout.addWidget(self.chat_area)  # Aggiunge l'area della chat al layout

    def setup_input_area(self):
        input_layout = QHBoxLayout()            # Layout orizzontale per l'input
        self.input_field = QLineEdit()          # Campo di testo per l'input dell'utente
        self.input_field.returnPressed.connect(self.send_message)  # Invio messaggio alla pressione di Invio
        input_layout.addWidget(self.input_field)  # Aggiunge il campo di testo al layout

        self.send_button = QPushButton("Send")    # Bottone per inviare il messaggio
        self.send_button.clicked.connect(self.send_message)  # Connessione del bottone all'invio
        input_layout.addWidget(self.send_button)   # Aggiunge il bottone al layout

        self.mic_button = QPushButton("🎤")        # Bottone per registrare audio
        self.mic_button.clicked.connect(self.toggle_recording)  # Connessione del bottone alla registrazione
        input_layout.addWidget(self.mic_button)    # Aggiunge il bottone al layout

        self.chat_layout.addLayout(input_layout)    # Aggiunge il layout dell'input all'area della chat

    def setup_sidebar(self):
        self.sidebar_layout.addWidget(QLabel("PrePrompt"))  # Etichetta per il prompt

        self.prompt_text = QTextEdit()             # Area di testo per modificare il prompt
        self.prompt_text.textChanged.connect(self.update_prompt)  # Connessione alla modifica del prompt
        self.sidebar_layout.addWidget(self.prompt_text)  # Aggiunge l'area del prompt alla sidebar

        self.restore_prompt_button = QPushButton("Restore Prompt")  # Bottone per ripristinare il prompt originale
        self.restore_prompt_button.clicked.connect(self.restore_prompt)  # Connessione del bottone al ripristino
        self.sidebar_layout.addWidget(self.restore_prompt_button)  # Aggiunge il bottone alla sidebar

        self.clear_button = QPushButton("Clear Conversation")  # Bottone per cancellare la conversazione
        self.clear_button.clicked.connect(self.clear_conversation)  # Connessione del bottone alla cancellazione
        self.sidebar_layout.addWidget(self.clear_button)          # Aggiunge il bottone alla sidebar

    def load_llm(self):
        print("Loading Llama 3.1")
        self.llm = Llama(
            model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            **config["load_params"]
        )  # Carica il modello Llama con i parametri di configurazione
        print("Llama 3.1 Model Loaded")

    def load_whisper(self):
        print("Loading Whisper Model")
        current_time = time.time()
        self.whisper_model = WhisperModel(
            "medium",        # Dimensione del modello Whisper
            device="cuda",   # Utilizza CUDA per l'accelerazione GPU
            compute_type="float16"  # Tipo di calcolo
        )
        time_elapsed = time.time() - current_time
        print(f"Whisper Model Loaded in {time_elapsed:.2f}s")

        # Warm-up di Whisper
        print("Warming up Whisper model")
        warmup_file = "uploads/example_warmup.wav"
        if os.path.exists(warmup_file):
            current_time = time.time()
            _, _ = self.whisper_model.transcribe(warmup_file)  # Esegue una trascrizione di warm-up
            time_elapsed = time.time() - current_time
            print(f"Whisper Model Warmed up in {time_elapsed:.2f}s")
        else:
            print("Warmup file not found. Skipping Whisper warm-up.")

    def load_prompts(self):
        self.prompts['default'] = config["inference_params"]["pre_prompt"]  # Carica il prompt predefinito dal config
        self.original_prompts = self.prompts.copy()  # Salva il prompt originale per il ripristino
        self.prompt_text.setText(self.prompts[self.current_prompt])  # Imposta il testo del prompt nell'interfaccia

    def send_message(self):
        user_input = self.input_field.text()  # Ottiene il testo inserito dall'utente
        if user_input.strip():  # Controlla se l'input non è vuoto
            # Prepara i messaggi per il modello
            prompt = self.get_current_prompt()
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Verifica se il numero di token supera il limite
            if self.count_tokens(messages) > (MAX_CONTEXT_TOKENS - RESERVED_TOKENS):
                QMessageBox.warning(
                    self,
                    "Prompt troppo lungo",
                    "Il prompt è troppo lungo e non può essere elaborato."
                )
                return  # Esce dalla funzione senza pulire l'input field
                
            # Se il prompt è di lunghezza accettabile, visualizza il messaggio dell'utente
            self.display_user_message(user_input)
            # Aggiunge il messaggio dell'utente alla cronologia
            self.chat_history.append({"role": "user", "content": user_input})
            # Pulisce il campo di input
            self.input_field.clear()
            # Genera la risposta dell'assistente
            self.generate_response(user_input)

    def display_user_message(self, user_input):
        user_html = markdown.markdown(user_input)  # Converte il testo in HTML usando Markdown
        self.chat_area.append("")                   # Aggiunge uno spazio nell'area della chat
        cursor = self.chat_area.textCursor()       # Ottiene il cursore dell'area della chat
        cursor.movePosition(QTextCursor.End)        # Sposta il cursore alla fine
        cursor.insertHtml(user_html)               # Inserisce il messaggio dell'utente
        block_format = QTextBlockFormat()
        block_format.setAlignment(Qt.AlignRight)    # Allinea il testo a destra (messaggio utente)
        cursor.setBlockFormat(block_format)        # Applica il formato
        self.chat_area.setTextCursor(cursor)        # Aggiorna il cursore nell'area della chat
        self.chat_area.append("<br>")               # Aggiunge una riga vuota

    def generate_response(self, user_input):
        response = self.generate_response_simple(user_input)  # Genera la risposta
        self.chat_history.append({"role": "assistant", "content": response})  # Aggiunge la risposta alla cronologia

    def generate_response_simple(self, user_input):
        prompt = self.get_current_prompt()
        messages = [
            {"role": "system", "content": prompt}
        ] + self.chat_history + [
            {"role": "user", "content": user_input}
        ]  # Prepara i messaggi per il modello

        # Gestione del limite di token
        while self.count_tokens(messages) > (MAX_CONTEXT_TOKENS - RESERVED_TOKENS):
            if len(self.chat_history) > 1: # Se la cronologia della chat ha più di un messaggio
                self.chat_history.pop(0)  # Rimuove il messaggio più vecchio
                messages = [
                    {"role": "system", "content": prompt}
                ] + self.chat_history + [
                    {"role": "user", "content": user_input}
                ]  # Aggiorna la lista messages con la cronologia aggiornata
            else:
                break  # Esce se non ci sono più messaggi da rimuovere


        # Inizializza il buffer della risposta dell'assistente
        self.assistant_response_buffer = ""
        self.chat_area.append("")  # Aggiunge uno spazio nell'area della chat
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.assistant_start_pos = cursor.position()  # Memorizza la posizione di inizio della risposta

        # Genera la risposta dell'assistente
        full_response = ""
        for token in self.llm.create_chat_completion(
            messages=messages,
            max_tokens=MAX_RESPONSE_TOKENS,
            stream=True
        ):
            token_text = token['choices'][0]['delta'].get('content', '')  # Ottiene il testo del token
            full_response += token_text  # Aggiunge il token alla risposta completa
            self.update_chat_area(token_text)  # Aggiorna l'area della chat con il nuovo token

        # Aggiungi uno spazio dopo la risposta
        self.chat_area.append("<br><br>")
        
        # Sostituisci il testo con il markdown formattato
        assistant_html = markdown.markdown(self.assistant_response_buffer)  # Converte la risposta in HTML
        cursor = self.chat_area.textCursor()
        cursor.setPosition(self.assistant_start_pos)  # Sposta il cursore all'inizio della risposta
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)  # Seleziona fino alla fine
        cursor.removeSelectedText()  # Rimuove il testo selezionato
        cursor.insertHtml(assistant_html + "<br><br>")  # Inserisce la risposta formattata
        block_format = QTextBlockFormat()
        block_format.setAlignment(Qt.AlignLeft)  # Allinea il testo a sinistra (risposta assistente)
        cursor.setBlockFormat(block_format)      # Applica il formato
        self.chat_area.setTextCursor(cursor)      # Aggiorna il cursore nell'area della chat
        self.chat_area.ensureCursorVisible()      # Assicura che il cursore sia visibile

        self.assistant_response_buffer = ""  # Resetta il buffer della risposta

        return full_response.strip()  # Ritorna la risposta completa senza spazi iniziali/finali

    def update_chat_area(self, new_token):
        self.assistant_response_buffer += new_token  # Aggiunge il nuovo token al buffer

        # Applica il markdown alla risposta corrente dell'assistente
        formatted_response = markdown.markdown(self.assistant_response_buffer)  # Converte in HTML

        # Posiziona il cursore all'inizio della risposta dell'assistente
        cursor = self.chat_area.textCursor()
        cursor.setPosition(self.assistant_start_pos)
        cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)

        # Sostituisce il testo corrente con il testo formattato
        cursor.removeSelectedText()  # Rimuove il testo selezionato
        cursor.insertHtml(formatted_response)  # Inserisce il testo formattato
        block_format = QTextBlockFormat()
        block_format.setAlignment(Qt.AlignLeft)  # Allinea a sinistra
        cursor.setBlockFormat(block_format)      # Applica il formato
        self.chat_area.setTextCursor(cursor)      # Aggiorna il cursore
        self.chat_area.ensureCursorVisible()      # Assicura che il cursore sia visibile
        QApplication.processEvents()              # Processa gli eventi dell'applicazione

    def toggle_recording(self):
        if not self.audio_recorder.is_recording:
            self.audio_recorder.start()           # Avvia la registrazione
            self.mic_button.setText("⏹")          # Cambia l'icona del bottone
        else:
            self.audio_recorder.stop()            # Ferma la registrazione
            self.mic_button.setText("🎤")          # Ripristina l'icona del bottone
    
    def on_transcription_finished(self, transcription):
        self.input_field.setText(transcription)  # Imposta la trascrizione nel campo di input
        self.send_message()                      # Invia il messaggio automaticamente

    def update_prompt(self):
        self.prompts[self.current_prompt] = self.prompt_text.toPlainText()  # Aggiorna il prompt corrente

    def restore_prompt(self):
        self.prompts[self.current_prompt] = self.original_prompts[self.current_prompt]  # Ripristina il prompt originale
        self.prompt_text.setText(self.prompts[self.current_prompt])  # Aggiorna l'interfaccia con il prompt ripristinato

    def clear_conversation(self):
        self.chat_area.clear()         # Cancella l'area della chat
        self.chat_history.clear()      # Cancella la cronologia della chat

    def get_current_prompt(self):
        return self.prompts[self.current_prompt]  # Ritorna il prompt corrente

    def count_tokens(self, messages):
        encoding = tiktoken.get_encoding("cl100k_base")  # Ottiene il metodo di encoding dei token
        num_tokens = 0
        for message in messages:
            num_tokens += len(encoding.encode(message['content']))  # Conta i token di ogni messaggio
        return num_tokens  # Ritorna il numero totale di token

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")  # Ottiene il metodo di encoding dei token
        num_tokens = len(encoding.encode(string))       # Conta i token nella stringa
        return num_tokens                              # Ritorna il numero di token

if __name__ == "__main__":
    app = QApplication(sys.argv)                     # Crea l'applicazione PyQt
    app.setStyle(QStyleFactory.create('Fusion'))     # Imposta lo stile 'Fusion' per l'applicazione
    window = ChatbotWindow()                         # Crea la finestra del chatbot
    window.show()                                     # Mostra la finestra
    sys.exit(app.exec_())                             # Avvia il loop dell'applicazione
