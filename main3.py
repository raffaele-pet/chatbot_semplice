# Chatbot semplice con GUI Eel

import eel
from llama_cpp import Llama
from config.config_Meta_Llama_3_1_8B_Instruct_Q4_K_M import config
import time

# Inizializzazione di Eel
eel.init('web')

# Caricamento del modello Llama
print("Loading Llama 3.1")
llm = Llama(model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", **config["load_params"])
print("Llama 3.1 Model Loaded")

# Inizializzazione della cronologia della chat
chat_history = []

@eel.expose
def generate_response(user_input):
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
            eel.update_chat(token_text)()

    chat_history.append({"role": "assistant", "content": response_text})
    return response_text

if __name__ == '__main__':
    print("Loading the model and starting Eel app...")
    time.sleep(1)
    print("Llama model loaded. Starting application...")
    eel.start('index.html', size=(800, 600))