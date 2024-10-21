document.getElementById('chat-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const userInput = document.getElementById('user-input').value;
    if (!userInput) return;

    addMessageToChatBox('Tu', userInput);
    document.getElementById('user-input').value = '';  // Pulisce l'input

    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');
    
    const senderElement = document.createElement('strong');
    senderElement.textContent = 'Assistant: ';

    const textElement = document.createElement('span');
    messageElement.appendChild(senderElement);
    messageElement.appendChild(textElement);
    chatBox.appendChild(messageElement);
    
    chatBox.scrollTop = chatBox.scrollHeight;

    // Invia la richiesta al server per iniziare lo streaming della risposta
    const response = await fetch('/stream', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: userInput }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let done = false;
    
    while (!done) {
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        const chunk = decoder.decode(value, { stream: true });
        
        // Estrae il testo dal chunk e lo aggiunge alla UI
        if (chunk) {
            const parsedChunk = chunk.split('\n\n').map(line => line.replace(/^data: /, '')).join('');
            textElement.textContent += parsedChunk;
            chatBox.scrollTop = chatBox.scrollHeight;  // Scorri verso il basso
        }
    }
});

function addMessageToChatBox(sender, message) {
    const chatBox = document.getElementById('chat-box');
    const messageElement = document.createElement('div');
    messageElement.classList.add('message');

    const senderElement = document.createElement('strong');
    senderElement.textContent = `${sender}: `;

    const textElement = document.createElement('span');
    textElement.textContent = message;

    messageElement.appendChild(senderElement);
    messageElement.appendChild(textElement);
    chatBox.appendChild(messageElement);

    chatBox.scrollTop = chatBox.scrollHeight;  // Scorri verso il basso
}
