import { getThreads, createAndRunThread, sendMessage, startThread, loadChatHistory, sendMessageStream, sendMessageTextStream, createAndRunThreadStream, moveHandlerQhali, handleTermsAndConditionsDisplay } from './services.js';

// Variable global para la grabación
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

// Función para alternar el estado de grabación
function toggleRecording() {
    console.log("[Voice] Toggle recording - current state:", isRecording);
    
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Función para configurar la grabación de voz
function setupVoiceRecording() {
    // Verificar si ya existe el botón de micrófono
    if (document.getElementById('mic-button')) {
        console.log("[Voice] Voice recording UI already set up");
        return;
    }
    
    console.log("[Voice] Setting up voice recording UI");
    
    // Crear botón de micrófono
    const micButton = document.createElement('button');
    micButton.id = 'mic-button';
    micButton.type = 'button';
    micButton.className = 'btn-mic';
    micButton.innerHTML = '<i class="fas fa-microphone"></i>';
    micButton.title = 'Grabar mensaje de voz (clic para iniciar/detener)';
    
    // Añadir evento para alternar grabación
    micButton.addEventListener('click', (e) => {
        e.preventDefault();
        toggleRecording();
    });
    
    // Crear indicador de grabación
    const recordingIndicator = document.createElement('div');
    recordingIndicator.id = 'recording-indicator';
    recordingIndicator.className = 'recording-indicator';
    recordingIndicator.innerHTML = '<div class="pulse-dot"></div> <span>Grabando... (clic en micrófono para detener)</span>';
    recordingIndicator.style.display = 'none';
    
    // Crear área de vista previa de la transcripción
    const transcriptionPreview = document.createElement('div');
    transcriptionPreview.id = 'transcription-preview';
    transcriptionPreview.className = 'transcription-preview';
    transcriptionPreview.style.display = 'none';
    
    // Obtener el formulario del chat
    const chatForm = document.querySelector('.chat-form');
    
    // Añadir elementos al formulario del chat
    chatForm.appendChild(micButton);
    chatForm.appendChild(recordingIndicator);
    chatForm.appendChild(transcriptionPreview);
    
    console.log("[Voice] Voice recording UI setup complete");
}

// Función para iniciar la grabación
async function startRecording() {
    console.log("[Voice] Starting recording");
    
    try {
        // Solicitar permisos para acceder al micrófono con configuración de alta calidad
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                channelCount: 1,
                sampleRate: 44100
            }
        });
        
        // Mostrar indicador de grabación
        const recordingIndicator = document.getElementById('recording-indicator');
        recordingIndicator.style.display = 'block';
        
        // Cambiar apariencia del botón de micrófono
        const micButton = document.getElementById('mic-button');
        micButton.classList.add('recording');
        micButton.innerHTML = '<i class="fas fa-stop"></i>'; // Cambiar a ícono de detener
        micButton.title = 'Detener grabación';
        
        // Configurar el MediaRecorder con opciones específicas para compatibilidad con Whisper
        const options = { mimeType: 'audio/webm' };
        mediaRecorder = new MediaRecorder(stream, options);
        audioChunks = [];
        
        // Evento para recopilar datos de audio
        mediaRecorder.addEventListener('dataavailable', event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        });
        
        // Evento para cuando se detiene la grabación
        mediaRecorder.addEventListener('stop', processRecording);
        
        // Duración máxima de grabación (30 segundos)
        setTimeout(() => {
            if (isRecording) {
                stopRecording();
            }
        }, 30000);
        
        // Iniciar la grabación
        mediaRecorder.start(100); // Capturar datos cada 100ms para mejor fluidez
        isRecording = true;
        
        console.log("[Voice] Recording started with options:", options);
    } catch (error) {
        console.error("[Voice] Error accessing microphone:", error);
        alert("No se pudo acceder al micrófono. Verifica los permisos e intenta nuevamente.");
    }
}

// Función para detener la grabación
function stopRecording() {
    console.log("[Voice] Stopping recording");
    
    if (mediaRecorder && isRecording) {
        // Detener el MediaRecorder
        mediaRecorder.stop();
        isRecording = false;
        
        // Ocultar indicador de grabación
        document.getElementById('recording-indicator').style.display = 'none';
        
        // Restaurar apariencia del botón de micrófono
        const micButton = document.getElementById('mic-button');
        micButton.classList.remove('recording');
        micButton.innerHTML = '<i class="fas fa-microphone"></i>'; // Volver al ícono de micrófono
        micButton.title = 'Grabar mensaje de voz (clic para iniciar/detener)';
        
        // Mostrar indicador de procesamiento
        const transcriptionPreview = document.getElementById('transcription-preview');
        transcriptionPreview.style.display = 'block';
        transcriptionPreview.textContent = 'Procesando audio...';
        
        // Detener todas las pistas del stream
        if (mediaRecorder.stream) {
            mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        console.log("[Voice] Recording stopped, processing audio");
    }
}

// Función para procesar la grabación
async function processRecording() {
    console.log("[Voice] Processing recording");
    
    try {
        // Crear el Blob de audio con todos los fragmentos grabados
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        console.log("[Voice] Audio blob created:", audioBlob.size, "bytes, type:", audioBlob.type);
        
        // Si el blob es demasiado pequeño, probablemente no se grabó nada
        if (audioBlob.size < 1000) {
            throw new Error("La grabación es demasiado corta o vacía");
        }
        
        // Crear FormData para enviar al servidor
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'recording.webm');
        
        console.log("[Voice] Sending audio to Whisper API...");
        // Enviar el audio al servidor para transcripción
        //const response = await fetch('/api/chat/speech-to-text', {
        const response = await fetch('/api/chat/local-speech-to-text', {
            method: 'POST',
            body: formData
        });
        
        // Manejar errores HTTP
        if (!response.ok) {
            const errorText = await response.text().catch(() => "");
            console.error(`[Voice] API error (${response.status}):`, errorText);
            
            let errorMessage;
            if (response.status === 422) {
                errorMessage = "Error en el formato de audio. Verifica los parámetros de la solicitud.";
            } else if (response.status === 500) {
                errorMessage = "Error interno del servidor al procesar el audio.";
            } else {
                errorMessage = `Error del servidor: ${response.status}`;
            }
            throw new Error(errorMessage);
        }
        
        const data = await response.json();
        console.log("[Voice] Transcription received:", data);
        
        // Mostrar la transcripción
        const transcriptionPreview = document.getElementById('transcription-preview');
        if (data.text && data.text.trim() !== '') {
            transcriptionPreview.textContent = data.text;
            
            // Enviar automáticamente después de 3 segundos
            setTimeout(() => {
                if (transcriptionPreview.style.display !== 'none') {
                    // Poner la transcripción en el input de texto
                    document.getElementById('user-message').value = data.text;
                    
                    // Enviar el mensaje
                    document.getElementById('chat-form').dispatchEvent(new Event('submit', { cancelable: true }));
                    
                    // Ocultar la vista previa
                    transcriptionPreview.style.display = 'none';
                }
            }, 500);
        } else {
            transcriptionPreview.textContent = 'No se detectó ningún texto. Intenta de nuevo.';
            setTimeout(() => {
                transcriptionPreview.style.display = 'none';
            }, 1000);
        }
    } catch (error) {
        console.error("[Voice] Error processing recording:", error);
        
        // Mostrar error en la vista previa
        const transcriptionPreview = document.getElementById('transcription-preview');
        transcriptionPreview.textContent = error.message || 'Error al procesar el audio. Intenta de nuevo.';
        setTimeout(() => {
            transcriptionPreview.style.display = 'none';
        }, 1000);
    }
}

// Esperamos la acción de enviar un mensaje
document.getElementById('chat-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    // Obtener el mensaje del usuario
    const userMessage = document.getElementById('user-message').value;
    
    // Si el mensaje está vacío, no hacer nada
    if (!userMessage.trim()) return;
    
    // Ocultar la vista previa de transcripción si está visible
    const transcriptionPreview = document.getElementById('transcription-preview');
    if (transcriptionPreview) {
        transcriptionPreview.style.display = 'none';
    }
    
    // Mostrar el mensaje del usuario en el chat
    addMessageToChat(userMessage, 'user');
    
    // Limpiar el campo de entrada
    document.getElementById('user-message').value = '';

    try {
        // Obtener el hilo activo desde localStorage
        let threadId = localStorage.getItem('active_thread_id');
        
        // Elemento para mostrar el indicador de escritura
        const typingIndicator = document.createElement('div');
        typingIndicator.classList.add('bot-typing-indicator');
        typingIndicator.textContent = "Qhali está escribiendo...";
        document.getElementById('chat-history').appendChild(typingIndicator);
        
        // Elemento para mostrar la respuesta en streaming (oculto inicialmente)
        const botMessageDiv = document.createElement('div');
        botMessageDiv.classList.add('bot-message');
        botMessageDiv.style.display = 'none'; // Oculto hasta que llegue el primer texto
        document.getElementById('chat-history').appendChild(botMessageDiv);
        
        // Hacer scroll para mostrar el indicador
        const chatHistory = document.getElementById('chat-history');
        chatHistory.scrollTop = chatHistory.scrollHeight;
        
        // Variable para rastrear si es el primer evento de texto
        let isFirstTextEvent = true;
        // Variable para rastrear si ha enviado movimiento
        let hasMoved = false;
        
        // Función para manejar los deltas de texto con actualización inmediata
        const handleTextDelta = (text) => {
            // Si es el primer evento de texto, mostrar el div de mensaje y eliminar el indicador
            if (isFirstTextEvent) {
                typingIndicator.remove(); // Eliminar el indicador de escritura
                botMessageDiv.style.display = 'block'; // Mostrar el div de mensaje
                isFirstTextEvent = false;
            }
            
            // Agregar el nuevo texto al mensaje
            botMessageDiv.textContent += text;

            // Asegurar que el mensaje sea visible haciendo scroll hasta el final
            requestAnimationFrame(() => {
                chatHistory.scrollTop = chatHistory.scrollHeight;
            });
        };
        
        if (!threadId) {
            // Si no hay hilo activo, creamos uno nuevo con streaming
            console.log("Creando nuevo hilo con streaming...");
            await createAndRunThreadStream({
                message: userMessage,
                onThreadCreated: (threadInfo) => {
                    console.log("Nuevo hilo creado:", threadInfo);
                },
                onTextDelta: handleTextDelta,
                onAudio: (audio) => {
                    console.log("Reproduciendo audio",hasMoved);
                    
                    if (!hasMoved){
                        console.log("Analizando " + userMessage)
                        moveHandlerQhali(userMessage);
                        hasMoved=true;
                    }
                    
                    // Asegurar que el audio se reproduzca
                    audio.play().catch(err => {
                        console.error("Error al reproducir audio:", err);
                    });
                },
                onCompletion: () => {
                    console.log("Respuesta completada");
                    // Si no hubo respuesta, reemplazar el indicador con un mensaje
                    if (isFirstTextEvent) {
                        typingIndicator.remove();
                        botMessageDiv.style.display = 'block';
                        botMessageDiv.textContent = "No he podido generar una respuesta. Por favor, intenta de nuevo.";
                    } else {
                        // Verificar términos y condiciones cuando la respuesta está completa
                        handleTermsAndConditionsDisplay(botMessageDiv.textContent, botMessageDiv);
                    }
                }
            });
        } else {
            // Para usar versión con audio, cambiar a sendMessageStream
            console.log("Enviando mensaje a hilo existente:", threadId);
            
            // Elegir entre texto solo o texto con audio
            const useAudio = true; // Activar streaming con audio
            
            if (useAudio) {
                await sendMessageStream({
                    thread_id: threadId, 
                    message: userMessage,
                    onTextDelta: handleTextDelta,
                    onAudio: (audio) => {
                        console.log("Reproduciendo audio");

                        if (!hasMoved){
                            console.log("Analizando " + userMessage)
                            moveHandlerQhali(userMessage);
                            hasMoved=true;
                        }    

                        // Asegurar que el audio se reproduzca
                        audio.play().catch(err => {
                            console.error("Error al reproducir audio:", err);
                        });
                    },
                    onCompletion: () => {
                        console.log("Respuesta completada");
                        // Si no hubo respuesta, reemplazar el indicador con un mensaje
                        if (isFirstTextEvent) {
                            typingIndicator.remove();
                            botMessageDiv.style.display = 'block';
                            botMessageDiv.textContent = "No he podido generar una respuesta. Por favor, intenta de nuevo.";
                        } else {
                            // Verificar términos y condiciones cuando la respuesta está completa
                            handleTermsAndConditionsDisplay(botMessageDiv.textContent, botMessageDiv);
                        }
                    }
                });
            } else {
                await sendMessageTextStream({
                    thread_id: threadId, 
                    message: userMessage,
                    onTextDelta: handleTextDelta,
                    onCompletion: () => {
                        console.log("Respuesta completada");
                        // Si no hubo respuesta, reemplazar el indicador con un mensaje
                        if (isFirstTextEvent) {
                            typingIndicator.remove();
                            botMessageDiv.style.display = 'block';
                            botMessageDiv.textContent = "No he podido generar una respuesta. Por favor, intenta de nuevo.";
                        } else {
                            // Verificar términos y condiciones cuando la respuesta está completa
                            handleTermsAndConditionsDisplay(botMessageDiv.textContent, botMessageDiv);
                        }
                    }
                });
            }
        }
    } catch (error) {
        console.error('Error:', error);
        // En caso de error, mostrar mensaje de error
        addMessageToChat("Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, intenta nuevamente.", 'bot');
    }
}); 

// Función para añadir mensajes al chat
function addMessageToChat(message, role) {
    const chatHistory = document.getElementById('chat-history');
    
    const messageDiv = document.createElement('div');
    messageDiv.textContent = message;

    // Agregar clase dependiendo del rol (usuario o bot)
    if (role === 'user') {
        messageDiv.classList.add('user-message');
    } else {
        messageDiv.classList.add('bot-message');
        // Verificar términos y condiciones solo en mensajes del bot
        handleTermsAndConditionsDisplay(message, messageDiv);
    }

    // Agregar mensaje al historial del chat
    chatHistory.appendChild(messageDiv);

    // Desplazar el chat hacia abajo
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Cargar el hilo activo y su historial al cargar la página
document.addEventListener('DOMContentLoaded', async () => {
    try {
        // Configurar la grabación de voz
        setupVoiceRecording();
        
        // Obtener los hilos existentes
        const threads = await getThreads();

        // Si hay hilos, obtener el más reciente
        let threadId = localStorage.getItem('active_thread_id');
        if (!threadId && threads.length > 0) {
            threadId = threads[0].id_thread;
            localStorage.setItem('active_thread_id', threadId); // Guardar el ID del hilo activo
        }

        if (threadId) {
            // Cargar el historial del hilo activo
            const history = await loadChatHistory(threadId);
            
            // Procesar cada mensaje del historial
            history.forEach(msg => {
                // Crear el div del mensaje
                const messageDiv = document.createElement('div');
                messageDiv.textContent = msg.message;
                messageDiv.classList.add(msg.role === 'user' ? 'user-message' : 'bot-message');
                
                // Agregar el mensaje al chat
                document.getElementById('chat-history').appendChild(messageDiv);
                
                // Si es un mensaje del bot, verificar términos y condiciones
                if (msg.role === 'bot' || msg.role === 'assistant') {
                    handleTermsAndConditionsDisplay(msg.message, messageDiv);
                }
            });

            // Hacer scroll al final del chat
            const chatHistory = document.getElementById('chat-history');
            chatHistory.scrollTop = chatHistory.scrollHeight;

            if (history.length < 1) {
                // Mostrar mensaje de bienvenida al chat si no hay historial
                addMessageToChat('Este es el inicio de tu conversación con Qhali', 'bot');
            }
        } else {
            // Si no hay hilo activo, mostrar mensaje de bienvenida
            if (history.length < 1) {
                // Mostrar mensaje de bienvenida al chat si no hay historial
                addMessageToChat('Este es el inicio de tu conversación con Qhali', 'bot');
            }
        }
    } catch (error) {
        console.error('Error al cargar los hilos o el historial:', error);
    }
});

// Función para crear un nuevo hilo (Nueva consulta)
document.getElementById('new-consultation').addEventListener('click', async () => {
    try {
        // Limpiar el historial de chat
        document.getElementById('chat-history').innerHTML = '';

        // Mostrar mensaje de bienvenida para el nuevo hilo
        const data = await startThread();
        
        // Guardar el ID del nuevo hilo
        localStorage.setItem('active_thread_id', data.id_thread);
        
        // Mostrar mensaje de bienvenida al nuevo hilo
		if (history.length < 1) {
                // Mostrar mensaje de bienvenida al chat si no hay historial
            addMessageToChat('Este es el inicio de tu conversación con Qhali', 'bot');
        }
    } catch (error) {
        console.error('Error al crear un nuevo hilo:', error);
    }
});
