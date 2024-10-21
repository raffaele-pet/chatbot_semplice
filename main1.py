# Chatbot semplice che funziona da terminale

# Importazione delle librerie necessarie
from llama_cpp import Llama  # Libreria per interagire con il modello LLM
from config.config_Meta_Llama_3_1_8B_Instruct_Q4_K_M import config  # Importazione della configurazione del modello

# Caricamento del modello LLM
print("Loading Llama 3.1")
# Inizializzazione del modello con i parametri specificati nella configurazione
llm = Llama(model_path="models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", **config["load_params"])
print("Llama 3.1 Model Loaded")

# Inizializzazione della cronologia della chat
chat_history = []

def generate_response(user_input):
    global chat_history
    # Aggiunge l'input dell'utente alla cronologia della chat
    chat_history.append({"role": "user", "content": user_input})

    # Prepara i messaggi per il modello, includendo il prompt di sistema e la cronologia della chat
    messages = [{"role": "system", "content": config["inference_params"]["pre_prompt"]}] + chat_history

    response_text = ""
    # Genera la risposta in modo incrementale (streaming)
    for token in llm.create_chat_completion(
        messages=messages,
        max_tokens=500,  # Limite massimo di token per la risposta
        stream=True  # Abilita lo streaming della risposta
    ):
        # Estrae il testo del token dalla risposta del modello
        token_text = token['choices'][0]['delta'].get('content', '')
        response_text += token_text
        # Stampa il token immediatamente, senza andare a capo
        print(token_text, end='', flush=True)

    print()  # Nuova riga dopo la risposta completa
    # Aggiunge la risposta del modello alla cronologia della chat
    chat_history.append({"role": "assistant", "content": response_text})

def main():
    print("Benvenuto nel chatbot. Digita 'exit' per uscire.")
    while True:
        # Richiede l'input dell'utente
        user_input = input("\nTu: ")
        # Controlla se l'utente vuole uscire
        if user_input.lower() == 'exit':
            break
        # Genera e stampa la risposta
        generate_response(user_input)

# Assicura che il codice venga eseguito solo se lo script è eseguito direttamente
if __name__ == "__main__":
    main()