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

    // Chiamata alla funzione Python
    await eel.generate_response(userInput);
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

    chatBox.scrollTop = chatBox.scrollHeight;
}

// Funzione esposta a Python per aggiornare la chat
eel.expose(update_chat);
function update_chat(token) {
    const chatBox = document.getElementById('chat-box');
    const lastMessage = chatBox.lastElementChild;
    if (lastMessage && lastMessage.querySelector('strong').textContent === 'Assistant: ') {
        lastMessage.querySelector('span').textContent += token;
    } else {
        addMessageToChatBox('Assistant', token);
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}