# Chatbot semplice con GUI Flask

from flask import Flask, render_template, request, Response
from llama_cpp import Llama
from config.config_Meta_Llama_3_1_8B_Instruct_Q4_K_M import config
import time

app = Flask(__name__)

# Caricamento del modello Llama
print("Loading Llama 3.1")
llm = Llama(model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", **config["load_params"])
print("Llama 3.1 Model Loaded")

# Inizializzazione della cronologia della chat
chat_history = []

def generate_response_stream(user_input):
    global chat_history
    chat_history.append({"role": "user", "content": user_input})

    messages = [{"role": "system", "content": config["inference_params"]["pre_prompt"]}] + chat_history
    
    response_text = ""
    for token in llm.create_chat_completion(
        messages=messages,
        max_tokens=500,
        stream=True
    ):
        token_text = token['choices'][0]['delta'].get('content', '')
        if token_text:
            response_text += token_text
            yield f"data: {token_text}\n\n"

    chat_history.append({"role": "assistant", "content": response_text})

# Rotta principale per la pagina web
@app.route('/')
def index():
    return render_template('index.html')

# Rotta per gestire il flusso di risposta
@app.route('/stream', methods=['POST'])
def stream():
    user_input = request.json['message']
    return Response(generate_response_stream(user_input), content_type='text/event-stream')

if __name__ == '__main__':
    print("Loading the model and starting Flask app...")
    # Aggiunge un ritardo per dare tempo di leggere il messaggio precedente
    time.sleep(1)
    print("Llama model loaded. Starting server...")
    # Avvio del server dopo il caricamento del modello
    app.run()
    print("Server running on http://127.0.0.1:5000")  # Questo apparirà dopo il caricamento del modello
