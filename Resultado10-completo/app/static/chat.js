import { sendMessageRagAdvancedStream, createThread, getThreadHistory, createRating } from './services.js';

// ========== GESTIÓN DE MENSAJES ==========

// Esperamos la acción de enviar un mensaje
document.getElementById('chat-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const userMessage = document.getElementById('user-message').value;
    
    if (!userMessage.trim()) return;
    
    // Mostrar el mensaje del usuario en el chat
    addMessageToChat(userMessage, 'user');
    
    // Limpiar el campo de entrada
    document.getElementById('user-message').value = '';

    try {
        let threadId = localStorage.getItem('active_thread_id');
        
        // Mostrar indicador de escritura
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('bot-typing-indicator');
        typingIndicator.textContent = "El agente está escribiendo...";
        document.getElementById('chat-history').appendChild(typingIndicator);
        
        // Elemento para la respuesta
        const botMessageDiv = document.createElement('div');
        botMessageDiv.classList.add('bot-message');
        botMessageDiv.style.display = 'none';
        document.getElementById('chat-history').appendChild(botMessageDiv);
        
        // Hacer scroll
        const chatHistory = document.getElementById('chat-history');
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        let isFirstTextEvent = true;
        let currentUserMsgId = null;
        let currentAssistantMsgId = null;
        
        // Si no hay hilo, crear uno nuevo
        if (!threadId) {
            console.log("Creando nuevo hilo...");
            const threadData = await createThread();
            threadId = threadData.id_thread;
            localStorage.setItem('active_thread_id', threadId);
            console.log("Nuevo hilo creado:", threadId);
        }
        
        // Enviar mensaje con RAG Advanced
        await sendMessageRagAdvancedStream({
            thread_id: threadId,
            message: userMessage,
            onSources: (sources) => {
                console.log("Mostrando fuentes:", sources);
                displaySources(sources);
            },
            onTextDelta: (text) => {
                // Mostrar mensaje en primer delta
                if (isFirstTextEvent) {
                    typingIndicator.remove();
                    botMessageDiv.style.display = 'block';
                    isFirstTextEvent = false;
                }
                
                botMessageDiv.textContent += text;
                
                // Auto-scroll
                requestAnimationFrame(() => {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                });
            },
            onCompletion: (data) => {
                console.log("Respuesta completada", data);
                
                // Guardar IDs para rating
                currentUserMsgId = data.user_msg_id;
                currentAssistantMsgId = data.assistant_msg_id;
                
                // Mostrar controles de rating
                if (currentUserMsgId && currentAssistantMsgId) {
                    displayRatingControls(
                        botMessageDiv,
                        currentUserMsgId,
                        currentAssistantMsgId
                    );
                }
                
                // Si no hubo respuesta
                if (isFirstTextEvent) {
                    typingIndicator.remove();
                    botMessageDiv.style.display = 'block';
                    botMessageDiv.textContent = "No he podido generar una respuesta.";
                }
            }
        });
    } catch (error) {
        console.error('Error:', error);
        addMessageToChat("Ha ocurrido un error. Por favor, intenta nuevamente.", 'bot');
    }
}); 

// Función para añadir mensajes al chat
function addMessageToChat(message, role) {
    const chatHistory = document.getElementById('chat-history');
    
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;
    messageDiv.classList.add(role === 'user' ? 'user-message' : 'bot-message');

    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// ========== FUENTES ==========

function displaySources(sources) {
    if (!sources || sources.length === 0) {
        console.log("[Sources] No sources to display");
        return;
    }
    
    console.log(`[Sources] Displaying ${sources.length} documents in side panel`);
    
    // Obtener panel y contenido
    const sourcesPanel = document.getElementById('sources-panel');
    const sourcesContent = document.getElementById('sources-content');
    
    // Limpiar contenido anterior
    sourcesContent.innerHTML = '';
    
    // Mostrar panel lateral
    sourcesPanel.classList.remove('hidden');
    
    // Crear cards para cada documento
    sources.forEach((doc, docIndex) => {
        const docCard = document.createElement('div');
        docCard.className = 'document-card';
        
        // Título del documento
        const docTitle = document.createElement('div');
        docTitle.className = 'doc-title';
        
        if (doc.titulo && doc.titulo.trim()) {
            docTitle.textContent = `${docIndex + 1}. ${doc.titulo}`;
        } else {
            docTitle.textContent = `${docIndex + 1}. ${doc.pdf_name}`;
        }
        docCard.appendChild(docTitle);
        
        // Enlace y estadísticas
        const docMeta = document.createElement('div');
        docMeta.className = 'doc-meta';
        
        // Enlace al documento
        const docLink = document.createElement('span');
        if (doc.fuente && doc.fuente.trim()) {
            const isValidUrl = doc.fuente.startsWith('http://') || doc.fuente.startsWith('https://');
            if (isValidUrl) {
                docLink.innerHTML = `📄 <a href="${doc.fuente}" target="_blank" rel="noopener noreferrer" class="doc-link">Ver documento</a>`;
            } else {
                docLink.textContent = `📄 ${doc.fuente}`;
            }
        }
        
        // Score promedio del documento
        const docScore = document.createElement('span');
        docScore.className = 'doc-score';
        const avgPercentage = (doc.avg_score * 100).toFixed(1);
        docScore.innerHTML = `<strong>${avgPercentage}%</strong>`;
        
        docMeta.appendChild(docLink);
        docMeta.appendChild(docScore);
        docCard.appendChild(docMeta);
        
        // Lista de chunks
        if (doc.chunks && doc.chunks.length > 0) {
            const chunksTitle = document.createElement('div');
            chunksTitle.className = 'chunks-title';
            chunksTitle.textContent = `Fragmentos relevantes (${doc.chunks.length}):`;
            docCard.appendChild(chunksTitle);
            
            // Listar cada chunk
            doc.chunks.forEach((chunk, chunkIndex) => {
                const chunkItem = document.createElement('div');
                chunkItem.className = 'chunk-item';
                
                // Header del chunk (índice + score)
                const chunkHeader = document.createElement('div');
                chunkHeader.className = 'chunk-header';
                
                const chunkInfo = document.createElement('span');
                chunkInfo.textContent = `📄 Fragmento #${chunk.chunk_index}`;
                
                const chunkScore = document.createElement('span');
                chunkScore.className = 'chunk-score';
                const chunkPercentage = (chunk.score * 100).toFixed(1);
                chunkScore.textContent = `${chunkPercentage}%`;
                
                chunkHeader.appendChild(chunkInfo);
                chunkHeader.appendChild(chunkScore);
                chunkItem.appendChild(chunkHeader);
                
                // Texto del chunk (si existe)
                if (chunk.text && chunk.text.trim()) {
                    const chunkText = document.createElement('div');
                    chunkText.className = 'chunk-text';
                    chunkText.textContent = chunk.text;
                    chunkItem.appendChild(chunkText);
                }
                
                docCard.appendChild(chunkItem);
            });
        }
        
        sourcesContent.appendChild(docCard);
    });
}

// ========== RATINGS ==========

function displayRatingControls(messageDiv, userMsgId, assistantMsgId) {
    const ratingContainer = document.createElement('div');
    ratingContainer.className = 'rating-container';
    ratingContainer.style.cssText = `
        margin-top: 10px;
        padding: 8px;
        background-color: #f9f9f9;
        border-radius: 4px;
        text-align: center;
    `;
    
    const ratingLabel = document.createElement('div');
    ratingLabel.textContent = '¿Qué tan útil fue esta respuesta?';
    ratingLabel.style.cssText = `
        font-size: 0.9em;
        color: #666;
        margin-bottom: 8px;
    `;
    ratingContainer.appendChild(ratingLabel);
    
    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = 'display: flex; gap: 5px; justify-content: center;';
    
    // Crear botones de rating (1-10)
    for (let i = 1; i <= 10; i++) {
        const btn = document.createElement('button');
        btn.textContent = i;
        btn.className = 'rating-button';
        btn.style.cssText = `
            padding: 5px 10px;
            border: 1px solid #ddd;
            background-color: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.2s;
        `;
        
        btn.addEventListener('mouseover', () => {
            btn.style.backgroundColor = '#4CAF50';
            btn.style.color = 'white';
            btn.style.borderColor = '#4CAF50';
        });
        
        btn.addEventListener('mouseout', () => {
            btn.style.backgroundColor = 'white';
            btn.style.color = 'black';
            btn.style.borderColor = '#ddd';
        });
        
        btn.addEventListener('click', async () => {
            try {
                await submitRating(userMsgId, assistantMsgId, i);
                ratingContainer.innerHTML = `
                    <div style="color: #4CAF50; font-size: 0.9em;">
                        ✓ Gracias por tu calificación: ${i}/10
                    </div>
                `;
            } catch (error) {
                console.error('Error al enviar rating:', error);
                ratingContainer.innerHTML = `
                    <div style="color: #f44336; font-size: 0.9em;">
                        ✗ Error al guardar la calificación
                    </div>
                `;
            }
        });
        
        buttonContainer.appendChild(btn);
    }
    
    ratingContainer.appendChild(buttonContainer);
    
    // Insertar después del mensaje
    messageDiv.insertAdjacentElement('afterend', ratingContainer);
}

async function submitRating(userMsgId, assistantMsgId, score) {
    console.log(`Enviando rating: user=${userMsgId}, assistant=${assistantMsgId}, score=${score}`);
    
    const result = await createRating({
        user_msg_id: userMsgId,
        assistant_msg_id: assistantMsgId,
        score: score
    });
    
    console.log('Rating guardado:', result);
    return result;
}

// ========== MODAL DE CONSENTIMIENTO ==========

function showConsentModal() {
    const modal = document.getElementById('consent-modal');
    modal.classList.add('show');
    
    // Deshabilitar scroll del body cuando modal está abierto
    document.body.style.overflow = 'hidden';
}

function hideConsentModal() {
    const modal = document.getElementById('consent-modal');
    modal.classList.remove('show');
    
    // Rehabilitar scroll del body
    document.body.style.overflow = 'auto';
}

function checkConsentGiven() {
    return localStorage.getItem('consent_given') === 'true';
}

function saveConsent() {
    localStorage.setItem('consent_given', 'true');
}

// ========== CARGA INICIAL ==========

document.addEventListener('DOMContentLoaded', async () => {
    // Event listener para checkbox de consentimiento
    const consentCheck = document.getElementById('consent-check');
    const consentAccept = document.getElementById('consent-accept');
    
    consentCheck.addEventListener('change', () => {
        consentAccept.disabled = !consentCheck.checked;
    });
    
    // Event listener para botón aceptar
    consentAccept.addEventListener('click', () => {
        saveConsent();
        hideConsentModal();
    });
    
    // Verificar si ya se dio consentimiento
    const consentGiven = checkConsentGiven();
    
    if (!consentGiven) {
        // Mostrar modal si no ha aceptado el consentimiento
        showConsentModal();
    }
    
    try {
        let threadId = localStorage.getItem('active_thread_id');
        
        if (threadId) {
            // Cargar el historial del hilo activo
            const history = await getThreadHistory(threadId);
            
            history.forEach(msg => {
                addMessageToChat(msg.message, msg.role);
            });

            const chatHistory = document.getElementById('chat-history');
            chatHistory.scrollTop = chatHistory.scrollHeight;

            if (history.length < 1) {
                addMessageToChat('Hola, soy tu Agente de IA de salud. ¿En qué puedo ayudarte hoy?', 'bot');
            }
        } else {
            // Mensaje de bienvenida
            addMessageToChat('Hola, soy tu Agente de IA de salud. ¿En qué puedo ayudarte hoy?', 'bot');
        }
    } catch (error) {
        console.error('Error al cargar el historial:', error);
    }
    
    // Event listener para cerrar panel de fuentes
    document.getElementById('close-sources').addEventListener('click', () => {
        const sourcesPanel = document.getElementById('sources-panel');
        sourcesPanel.classList.add('hidden');
    });
});

// ========== NUEVA CONSULTA ==========

document.getElementById('new-consultation').addEventListener('click', async () => {
    try {
        // Mostrar modal de consentimiento para nueva consulta
        // Resetear checkbox y botón
        document.getElementById('consent-check').checked = false;
        document.getElementById('consent-accept').disabled = true;
        showConsentModal();
        
        // Esperar a que se acepte el consentimiento
        const waitForConsent = new Promise((resolve) => {
            const checkInterval = setInterval(() => {
                const modal = document.getElementById('consent-modal');
                if (!modal.classList.contains('show')) {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);
        });
        
        await waitForConsent;
        
        // Limpiar el chat
        document.getElementById('chat-history').innerHTML = '';

        // Crear nuevo hilo
        const threadData = await createThread();
        localStorage.setItem('active_thread_id', threadData.id_thread);
        
        // Mensaje de bienvenida
        addMessageToChat('Hola, soy tu Agente de IA de salud. ¿En qué puedo ayudarte hoy?', 'bot');
        
        console.log('Nuevo hilo creado:', threadData.id_thread);
    } catch (error) {
        console.error('Error al crear nuevo hilo:', error);
    }
});
