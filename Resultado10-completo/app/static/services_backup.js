// services.js
import { audioService } from './audio-service.js';


// Función para obtener todos los hilos de chat
export async function getThreads() {
    try {
        const response = await fetch('/api/chat/threads',{
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Error al obtener los hilos');
        }
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Función para crear un nuevo hilo de chat y enviar el primer mensaje
export async function createAndRunThread(message) {
    try {
        const response = await fetch('/api/chat/threads/create-run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });

        if (!response.ok) {
            throw new Error('Error al crear el hilo');
        }

        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Función para enviar un mensaje al chat en un hilo existente
export async function sendMessage({ thread_id, message }) {
    try {
        const response = await fetch('/api/chat/send', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({thread_id: thread_id, message: message })
        });

        if (!response.ok) {
            throw new Error('Error al enviar el mensaje');
        }

        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

export async function sendMessageWithAudio({ thread_id, message }) {
    try {
        const response = await fetch('/api/chat/send-audio', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ thread_id: thread_id, message: message })
        });

        if (!response.ok) {
            throw new Error('Error al enviar el mensaje');
        }

        // Get the text from the header
        const textContent = response.headers.get('Text-Content');
        // Get the timing info from the header
        const timingInfo = JSON.parse(response.headers.get('Timing-Info') || '{}');
        
        // Get the audio blob directly
        const audioBlob = await response.blob();
        
        // Create an audio element and play the audio
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();

        return {
            message: textContent,
            timing: timingInfo
        };
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

export async function startThread(){
    try {
        const response = await fetch('/api/chat/threads/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Error al crear el hilo');
        }

        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }

};

export async function loadChatHistory (threadId){
    try {
        const response = await fetch(`/api/chat/history/${threadId}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
        });

        if (!response.ok) {
            throw new Error('Error al recuperar el hilo');
        }

        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
};

export async function sendMessageTextStream({ thread_id, message, onTextDelta, onCompletion }) {
    try {
        const response = await fetch('/api/chat/send-message-text-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache' // Importante para streaming
            },
            body: JSON.stringify({ thread_id: thread_id, message: message })
        });

        if (!response.ok) {
            throw new Error('Error al enviar el mensaje en streaming');
        }

        console.time();
        console.log("The first header was received");
        // Procesar la respuesta de streaming
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        // Función para procesar datos de streaming
        async function readStream() {
            console.timeEnd();
            console.time();
            console.log("Reading the stream");
            try {
                const { value, done } = await reader.read();
                
                if (done) {
                    // Procesar cualquier datos pendientes en el buffer
                    if (buffer.trim()) {
                        processEventData(buffer);
                    }
                    return;
                }
            
                // Decodificar y acumular datos
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
                
                // Procesar eventos completos en el buffer usando una expresión regular más robusta
                // Esta regex maneja diferentes tipos de saltos de línea y eventos SSE
                const eventRegex = /(?:data:\s*)?(\{.*?\})\s*(?:\n\n|\r\n\r\n|\r\r)/g;
                let match;
                let lastIndex = 0;

                // Buscar eventos completos en el buffer
                while ((match = eventRegex.exec(buffer)) !== null) {
                    const eventData = match[1];
                    if (eventData && !eventData.includes(': keep-alive')) {
                        await processEventData(eventData);
                    }
                    lastIndex = match.index + match[0].length;
                }

				if (buffer.length > 4000000) {
					console.warn('[Stream] Buffer overflow, clearing incomplete data');
					buffer = ''; // Limpiar el buffer
				}
				
                // Mantener solo la parte no procesada en el buffer
                buffer = buffer.slice(lastIndex);

                // Continuar leyendo inmediatamente
                requestAnimationFrame(() => readStream());
            } catch (error) {
                console.error('[Stream] Error en la lectura del stream:', error);
            }
        }
        
        async function processEventData(eventText) {
            try {
                // Limpiar el texto del evento de cualquier prefijo 'data:' residual
                eventText = eventText.replace(/^data:\s*/m, '').trim();
                
                // Ignorar explícitamente los eventos keep-alive
                if (eventText.includes(': keep-alive')) {
                    return;
                }

                const eventData = JSON.parse(eventText);
                
                // Manejar diferentes tipos de eventos
                switch (eventData.type) {
                    case 'text_delta':
                        if (onTextDelta) {
                            // Procesar texto inmediatamente sin esperar
                            onTextDelta(eventData.text);
                            // Forzar actualización del DOM
                            await new Promise(resolve => setTimeout(resolve, 0));
                        }
                        break;
                    case 'completion':
                        if (onCompletion) onCompletion(eventData);
                        break;
                    case 'error':
                        console.error('[Stream] Error del servidor:', eventData.message);
                        if (onTextDelta) onTextDelta("Lo siento, ha ocurrido un error. Por favor, intenta nuevamente.");
                        break;
                }
            } catch (e) {
                console.error('Error parsing SSE data:', e, eventText);
            }
        }
        
        // Iniciar el procesamiento del stream
        await readStream();
        
        return true;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

export async function sendMessageStream({ thread_id, message, onTextDelta, onAudio, onSources, onCompletion }) {
    try {
        console.log(`[sendMessageStream] Starting stream request for thread ${thread_id}`);
        const startTime = performance.now();
        console.time();
        // Inicia la solicitud de streaming usando RAG Advanced
        const response = await fetch('/api/chat/send-message-rag-advanced-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            },
            body: JSON.stringify({ thread_id: thread_id, message: message })
        });
        console.log("fetch recevied");
        console.timeEnd();

        console.log(`[sendMessageStream] Response received in ${performance.now() - startTime}ms`);
        
        if (!response.ok) {
            throw new Error('Error al enviar el mensaje en streaming');
        }

        // Procesar la respuesta de streaming
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let textDeltaCount = 0;

        console.time();
        console.log("Procesando stream");
        // Función para procesar datos de streaming
        async function readStream() {
            try {
                console.timeEnd();
                console.time();
                const { value, done } = await reader.read();
                
                if (done) {
                    console.log(`[Stream] Stream complete. Processed ${textDeltaCount} text deltas and queued ${audioChunkCount} audio chunks`);
                    // Procesar cualquier dato restante en el buffer
                    if (buffer.trim()) {
                        console.log("processing event data of remaining buffer");
                        await processEventData(buffer);
                    }
                    return;
                }
                
                // Decodificar y acumular datos
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
                
                // Procesar eventos completos en el buffer usando una expresión regular más robusta
                // Esta regex maneja diferentes tipos de saltos de línea y eventos SSE
                const eventRegex = /(?:data:\s*)?(\{.*?\})\s*(?:\n\n|\r\n\r\n|\r\r)/g;
                let match;
                let lastIndex = 0;

                // Buscar eventos completos en el buffer
                while ((match = eventRegex.exec(buffer)) !== null) {
                    const eventData = match[1];
                    if (eventData && !eventData.includes(': keep-alive')) {
                        console.log("processing event data of full buffer");

                        await processEventData(eventData);
                    }
                    lastIndex = match.index + match[0].length;
                }

                // Mantener solo la parte no procesada en el buffer
                buffer = buffer.slice(lastIndex);

                // Si el buffer es muy grande y no contiene un evento completo, limpiarlo
                if (buffer.length > 4000000) {
                    console.warn('[Stream] Buffer overflow, clearing incomplete data');
                    buffer = '';
                }
                
                // Continuar leyendo inmediatamente
                requestAnimationFrame(() => readStream());
            } catch (error) {
                console.error('[Stream] Error en la lectura del stream:', error);
            }
        }
        
        // Procesar cada evento SSE individual
        async function processEventData(eventText) {
            try {
                // Limpiar el texto del evento de cualquier prefijo 'data:' residual
                eventText = eventText.replace(/^data:\s*/m, '').trim();
                
                // Ignorar explícitamente los eventos keep-alive
                if (eventText.includes(': keep-alive')) {
                    return;
                }

                const eventData = JSON.parse(eventText);
                console.timeEnd();
                console.time();
                console.log(eventData.type);
                // Manejar diferentes tipos de eventos
                switch (eventData.type) {
                    case 'sources':
                        console.log(`[Stream] Sources received:`, eventData.sources);
                        if (onSources) {
                            onSources(eventData.sources);
                        }
                        break;
                    case 'text_delta':
                        textDeltaCount++;
                        if (textDeltaCount % 10 === 0) {
                            console.log(`[Stream] Received ${textDeltaCount} text deltas so far`);
                        }
                        if (onTextDelta) {
                            onTextDelta(eventData.text);
                            // Forzar actualización del DOM
                            await new Promise(resolve => setTimeout(resolve, 0));
                        }
                        break;
                    case 'text_with_audio':
                        // Siempre mostrar el texto inmediatamente
                        if (eventData.text) {
                            textDeltaCount++;
                            if (onTextDelta) onTextDelta(eventData.text);
                        }
                        
                        // Procesar el audio usando el servicio de audio
                        if (eventData.audio_hex) {
                            audioService.processAndQueueAudio(eventData.audio_hex, onAudio);
                        }
                        break;
                    case 'audio_bytes':
                        // Procesar el audio usando el servicio de audio
                        if (eventData.audio_hex) {
                            audioService.processAndQueueAudio(eventData.audio_hex, onAudio);
                        }
                        break;
                    case 'error':
                        console.error('[Stream] Error del servidor:', eventData.message);
                        if (onTextDelta) onTextDelta("Lo siento, ha ocurrido un error. Por favor, intenta nuevamente.");
                        break;
                    case 'completion':
                        console.log(`[Stream] Completion received. Total time: ${eventData.timing?.total_request_time || 'unknown'}ms`);
                        if (onCompletion) onCompletion(eventData);
                        break;
                }
            } catch (e) {
                console.error('[Stream] Error parsing SSE data:', e, eventText);
            }
        }
        
        console.log(`[sendMessageStream] Starting stream processing`);
        // Iniciar el procesamiento del stream
        await readStream();
        
        return true;
    } catch (error) {
        console.error('[sendMessageStream] Error:', error);
        throw error;
    }
}

//Manejadores de movimiento
function esRechazoTerminos(userInput) {
    const texto = userInput.toLowerCase();
    const patronesRechazo = [
        /\bno\s+(quiero|deseo|acepto|aceptaré|continuar|seguir)\b/,
        /\bno\s+voy\s+a\s+(aceptar|continuar|seguir)\b/,
        /\bme\s+niego(\s+a\s+(aceptar|continuar|seguir))?\b/,
        /\bno\s+(quiero|pienso|deseo)\b.*\b(términos?|condiciones?)\b/,
        /\bno\b.*\bacepto\b/,
        /\brechazo\b/,
        /\bdeclino\b/,
        /\bme\s+rehuso\b/,
        /\bno\s+acept[oé]?\b/,
        /\bni\s+acepto\b/,
        /\bni\s+quiero\b/,
    ];
    return patronesRechazo.some(p => p.test(texto));
}

function esFinConversacion(userInput) {
    const texto = userInput.toLowerCase();
    const patrones = [
        /\b(gracias|muchas gracias|okey|ya está|todo bien)\b.*\b(adiós|chau|terminar|cerrar|eso es todo|eso sería todo|hasta luego)\b/,
        /\bterminar\b.*\bconversación\b/,
        /\bcerrar\b.*\bchat\b/,
        /\bye\b|\badiós\b|\bhasta luego\b/,
        /\bno\b.*\b(tengo|necesito|más preguntas|más dudas)\b/,
        /\beso\b.*\b(nomás|sería todo|fue todo|era nomás|era todo)\b/,
        /\bterminé\b|\bya fue\b|\bya acabé\b/
    ];
    return patrones.some(p => p.test(texto));
}

function esPedidoAgente(userInput) {
    const texto = userInput.toLowerCase();
    const patrones = [
        /\bquiero\b.*\bagente\b/,
        /\bquiero\b.*\basesor\b/,
        /\bpuedo\b.*\bhablar\b.*\bhumano\b/,
        /\bnecesito\b.*\bpersona\b/,
        /\bquiero\b.*\bcontactar\b.*\balguien\b/,
        /\bme\b.*\bderiven?\b.*\bhumano\b/,
        /\bchat\b.*\bagente\b/,
        /\bchat\b.*\basesor\b/,
        /\bpuedo\b.*\bcomunicarme\b.*\b(una persona|un agente|un asesor|humano)\b/,
    ];
    return patrones.some(p => p.test(texto));
}

function esSaludo(userInput) {
    const texto = userInput.toLowerCase();
    const patrones = [
        /\bhola\b/,
        /\bholi(s)?\b/,
        /\bholita(s)?\b/,
        /\bhey\b/,
        /\bhello\b/,
        /\bhi\b/,
        /\bsaludos\b/,
        /\bqué tal\b/,
        /\bcomo (estás|estais|anda[s]?)\b/,
        /\bbuenos días\b/,
        /\bbuenas tardes\b/,
        /\bbuenas noches\b/,
        /\bqué onda\b/,
        /\bqué más\b/,
        /\bqué hay\b/,
        /\bqué fue\b/,
        /\bque xopa\b/,
        /\bque pasa\b/,
        /\bq tal\b/,
        /\bwenas\b/,
        /\bbuen día\b/,
        /\bgusto en saludarte\b/,
        /\bqué gusto\b/,
        /\bmuy buenas\b/,
        /\bmuy buen[oa]s?\b/,
    ];
    return patrones.some(p => p.test(texto));
}

export async function fireMovementQhali(action){
    console.log("firing movement");
    //const response = fetch('https://credible-clam-tolerant.ngrok-free.app/action/'+action,{
    const response = fetch('http://10.100.184.20:8080/action/'+action,{
        method: 'POST',
        headers: {
            'Content-Type': 'application/text',
        },
    }).then((res)=>{
        res.text()
    }).then((text)=> console.log(text));
}

export async function moveHandlerQhali(userMessage) {
    let action="";
    if (esRechazoTerminos(userMessage)) {
        action = "hn1-esp";
    } else if (esFinConversacion(userMessage) || esPedidoAgente(userMessage) || esSaludo(userMessage)) {
        action = "hn3-esp";
    } else {
        const opciones = ["hn4", "hn6"];
        action = opciones[Math.floor(Math.random() * opciones.length)] + "-esp";
    }
    console.log("moving qhali with", action)
    await fireMovementQhali(action);
}

export async function createAndRunThreadStream({ message, onThreadCreated, onTextDelta, onAudio, onCompletion }) {
    try {
        console.log(`[createAndRunThreadStream] Starting new thread stream`);
        const startTime = performance.now();
        
        // Inicia la solicitud de streaming
        const response = await fetch('/api/chat/threads/create-run-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/event-stream',
                'Cache-Control': 'no-cache'
            },
            body: JSON.stringify({ message: message })
        });

        console.log(`[createAndRunThreadStream] Response received in ${performance.now() - startTime}ms`);

        if (!response.ok) {
            throw new Error('Error al crear el hilo y enviar mensaje en streaming');
        }

        // Procesar la respuesta de streaming
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let threadInfo = null;
        let textDeltaCount = 0;

        // Función para procesar datos de streaming
        async function readStream() {
            try {
                const { value, done } = await reader.read();
                
                if (done) {
                    console.log(`[Stream] Stream complete. Processed ${textDeltaCount} text deltas and queued ${audioChunkCount} audio chunks`);
                    // Procesar cualquier datos pendientes en el buffer
                    if (buffer.trim()) {
                        await processEventData(buffer);
                    }
                    return;
                }
                
                // Decodificar y acumular datos
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
                
                // Procesar eventos completos en el buffer usando una expresión regular más robusta
                // Esta regex maneja diferentes tipos de saltos de línea y eventos SSE
                const eventRegex = /(?:data:\s*)?(\{.*?\})\s*(?:\n\n|\r\n\r\n|\r\r)/g;
                let match;
                let lastIndex = 0;

                // Buscar eventos completos en el buffer
                while ((match = eventRegex.exec(buffer)) !== null) {
                    const eventData = match[1];
                    if (eventData && !eventData.includes(': keep-alive')) {
                        await processEventData(eventData);
                    }
                    lastIndex = match.index + match[0].length;
                }

                // Mantener solo la parte no procesada en el buffer
                buffer = buffer.slice(lastIndex);

                // Si el buffer es muy grande y no contiene un evento completo, limpiarlo
                if (buffer.length > 4000000) {
                    console.warn('[Stream] Buffer overflow, clearing incomplete data');
                    buffer = '';
                }
                
                // Continuar leyendo inmediatamente
                requestAnimationFrame(() => readStream());
            } catch (error) {
                console.error('[Stream] Error en la lectura del stream:', error);
            }
        }
        
        // Procesar un evento individual
        async function processEventData(eventText) {
            try {
                // Limpiar el texto del evento de cualquier prefijo 'data:' residual
                eventText = eventText.replace(/^data:\s*/m, '').trim();
                
                // Ignorar explícitamente los eventos keep-alive
                if (eventText.includes(': keep-alive')) {
                    return;
                }

                const eventData = JSON.parse(eventText);
                
                // Manejar diferentes tipos de eventos
                switch (eventData.type) {
                    case 'thread_created':
                        threadInfo = {
                            id_thread: eventData.id_thread,
                            title: eventData.title
                        };
                        console.log(`[Thread] New thread created: ${eventData.id_thread}`);
                        if (onThreadCreated) onThreadCreated(threadInfo);
                        // Guardar el ID del hilo en localStorage
                        localStorage.setItem('active_thread_id', eventData.id_thread);
                        break;
                    case 'text_delta':
                        textDeltaCount++;
                        if (textDeltaCount % 10 === 0) {
                            console.log(`[Stream] Received ${textDeltaCount} text deltas so far`);
                        }
                        if (onTextDelta) {
                            onTextDelta(eventData.text);
                            // Forzar actualización del DOM
                            await new Promise(resolve => setTimeout(resolve, 0));
                        }
                        break;
                    case 'text_with_audio':
                        // Mostrar el texto inmediatamente 
                        if (eventData.text) {
                            textDeltaCount++;
                            if (onTextDelta) onTextDelta(eventData.text);
                        }
                        
                        // Procesar el audio usando el servicio de audio
                        if (eventData.audio_hex) {
                            audioService.processAndQueueAudio(eventData.audio_hex, onAudio);
                        }
                        break;
                    case 'audio_bytes':
                        // Procesar el audio usando el servicio de audio
                        if (eventData.audio_hex) {
                            audioService.processAndQueueAudio(eventData.audio_hex, onAudio);
                        }
                        break;
                    case 'error':
                        console.error('[Stream] Error del servidor:', eventData.message);
                        if (onTextDelta) onTextDelta("Lo siento, ha ocurrido un error. Por favor, intenta nuevamente.");
                        break;
                    case 'completion':
                        console.log(`[Stream] Completion received. Total time: ${eventData.timing?.total_request_time || 'unknown'}ms`);
                        if (onCompletion) onCompletion({ ...eventData, ...threadInfo });
                        break;
                }
            } catch (e) {
                console.error('[Stream] Error parsing SSE data:', e, eventText);
            }
        }
        
        console.log(`[createAndRunThreadStream] Starting stream processing`);
        // Iniciar el procesamiento del stream
        await readStream();
        
        return threadInfo;
    } catch (error) {
        console.error('[createAndRunThreadStream] Error:', error);
        throw error;
    }
}

// Función para generar TTS de cualquier texto
export async function textToSpeech(message) {
    try {
        console.log("Generando audio TTS para texto");
        
        const response = await fetch('/api/chat/text-to-speech', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });

        if (!response.ok) {
            throw new Error(`Error TTS: ${response.status}`);
        }

        // Convertir la respuesta a blob de audio
        const audioBlob = await response.blob();
        
        // Crear URL del audio y crear elemento de audio
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        console.log("Audio TTS generado exitosamente");
        
        // Limpiar la URL después de que termine la reproducción
        audio.addEventListener('ended', () => {
            URL.revokeObjectURL(audioUrl);
        });
        
        return audio;
        
    } catch (error) {
        console.error('Error al generar TTS:', error);
        return null;
    }
}

// Función para manejar la visualización de términos y condiciones
export function handleTermsAndConditionsDisplay(responseText, messageElement = null) {
    // Patrones para detectar menciones de términos y condiciones
    const termsPatterns = [
        "¿Deseas aceptarlos?",
        "aceptes nuestros términos y condiciones",
    ];

    // Verificar si alguno de los patrones está presente en el texto
    const shouldShowTerms = termsPatterns.some(pattern => responseText.includes(pattern));

    // Eliminar cualquier contenedor de términos existente para evitar duplicados
    const existingContainer = document.getElementById('terms-image-container');
    if (existingContainer) {
        existingContainer.remove();
    }

    if (shouldShowTerms) {
        // Crear nuevo contenedor
        const termsImageContainer = document.createElement('div');
        termsImageContainer.id = 'terms-image-container';
        termsImageContainer.className = 'terms-image-container';

        // Crear la imagen usando el archivo SVG
        const termsImage = document.createElement('img');
        termsImage.src = "/static/qr-qhali.svg"; // Ruta al archivo SVG
        termsImage.alt = "Términos y Condiciones";
        termsImage.style.maxWidth = "50%";
        termsImage.style.height = "auto";
        termsImage.style.margin = "10px 0";

        // Agregar la imagen al contenedor
        termsImageContainer.appendChild(termsImage);

        // Determinar dónde insertar el contenedor
        if (messageElement) {
            // Si se proporciona el elemento del mensaje, insertar después de él
            messageElement.insertAdjacentElement('afterend', termsImageContainer);
        } else {
            // Si no hay elemento de mensaje, buscar el último mensaje del bot
            const chatHistory = document.getElementById('chat-history');
            const botMessages = chatHistory.getElementsByClassName('bot-message');
            if (botMessages.length > 0) {
                const lastBotMessage = botMessages[botMessages.length - 1];
                lastBotMessage.insertAdjacentElement('afterend', termsImageContainer);
            } else {
                // Si no hay mensajes del bot, añadir al final del chat
                chatHistory.appendChild(termsImageContainer);
            }
        }
    }
} 
 