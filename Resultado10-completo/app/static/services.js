// ========== SERVICIOS SIMPLIFICADOS - Sin audio, sin OpenAI ==========

// Obtener todos los hilos
export async function getThreads() {
    try {
        const response = await fetch('/api/chat/threads', {
            method: 'GET',
            headers: {'Content-Type': 'application/json'},
        });

        if (!response.ok) throw new Error('Error al obtener los hilos');
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Crear un nuevo hilo
export async function createThread() {
    try {
        const response = await fetch('/api/chat/threads/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
        });

        if (!response.ok) throw new Error('Error al crear el hilo');
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Cargar historial de un hilo
export async function getThreadHistory(threadId) {
    try {
        const response = await fetch(`/api/chat/history/${threadId}`, {
            method: 'GET',
            headers: {'Content-Type': 'application/json'},
        });

        if (!response.ok) throw new Error('Error al recuperar el historial');
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Enviar mensaje con RAG Advanced y obtener streaming
export async function sendMessageRagAdvancedStream({ 
    thread_id, 
    message, 
    onSources, 
    onTextDelta, 
    onCompletion 
}) {
    try {
        console.log(`[sendMessageRagAdvancedStream] Starting for thread ${thread_id}`);
        const startTime = performance.now();
        
        const response = await fetch('/api/chat/send-message-rag-advanced-stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Cache-Control': 'no-cache'
            },
            body: JSON.stringify({ thread_id, message })
        });

        console.log(`[sendMessageRagAdvancedStream] Response in ${performance.now() - startTime}ms`);
        
        if (!response.ok) throw new Error('Error al enviar el mensaje');

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let textDeltaCount = 0;

        async function readStream() {
            try {
                const { value, done } = await reader.read();
                
                if (done) {
                    console.log(`[Stream] Complete. Processed ${textDeltaCount} text deltas`);
                    if (buffer.trim()) {
                        await processEventData(buffer);
                    }
                    return;
                }
                
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
                
                const eventRegex = /(?:data:\s*)?(\{.*?\})\s*(?:\n\n|\r\n\r\n|\r\r)/g;
                let match;
                let lastIndex = 0;

                while ((match = eventRegex.exec(buffer)) !== null) {
                    const eventData = match[1];
                    if (eventData && !eventData.includes(': keep-alive')) {
                        await processEventData(eventData);
                    }
                    lastIndex = match.index + match[0].length;
                }

                buffer = buffer.slice(lastIndex);

                if (buffer.length > 4000000) {
                    console.warn('[Stream] Buffer overflow, clearing');
                    buffer = '';
                }
                
                requestAnimationFrame(() => readStream());
            } catch (error) {
                console.error('[Stream] Error reading:', error);
            }
        }
        
        async function processEventData(eventText) {
            try {
                eventText = eventText.replace(/^data:\s*/m, '').trim();
                
                if (eventText.includes(': keep-alive')) return;

                const eventData = JSON.parse(eventText);
                
                switch (eventData.type) {
                    case 'sources':
                        console.log('[Stream] Sources received:', eventData.sources);
                        if (onSources) onSources(eventData.sources);
                        break;
                        
                    case 'text_delta':
                        textDeltaCount++;
                        if (textDeltaCount % 10 === 0) {
                            console.log(`[Stream] ${textDeltaCount} text deltas processed`);
                        }
                        if (onTextDelta) {
                            onTextDelta(eventData.text);
                            await new Promise(resolve => setTimeout(resolve, 0));
                        }
                        break;
                        
                    case 'completion':
                        console.log('[Stream] Completion received');
                        if (onCompletion) onCompletion(eventData);
                        break;
                        
                    case 'error':
                        console.error('[Stream] Server error:', eventData.message);
                        if (onTextDelta) {
                            onTextDelta("Lo siento, ha ocurrido un error.");
                        }
                        break;
                }
            } catch (e) {
                console.error('[Stream] Error parsing SSE:', e, eventText);
            }
        }
        
        await readStream();
        return true;
        
    } catch (error) {
        console.error('[sendMessageRagAdvancedStream] Error:', error);
        throw error;
    }
}

// Crear un rating para una interacción
export async function createRating({ user_msg_id, assistant_msg_id, score }) {
    try {
        console.log(`[createRating] user=${user_msg_id}, assistant=${assistant_msg_id}, score=${score}`);
        
        const response = await fetch('/api/chat/ratings/create', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ user_msg_id, assistant_msg_id, score })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Error al crear el rating');
        }
        
        const result = await response.json();
        console.log('[createRating] Success:', result);
        return result;
    } catch (error) {
        console.error('[createRating] Error:', error);
        throw error;
    }
}

// Obtener ratings de un hilo
export async function getThreadRatings(threadId) {
    try {
        const response = await fetch(`/api/chat/ratings/${threadId}`, {
            method: 'GET',
            headers: {'Content-Type': 'application/json'},
        });

        if (!response.ok) throw new Error('Error al obtener ratings');
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Obtener todos los ratings
export async function getAllRatings() {
    try {
        const response = await fetch('/api/chat/ratings', {
            method: 'GET',
            headers: {'Content-Type': 'application/json'},
        });

        if (!response.ok) throw new Error('Error al obtener ratings');
        return await response.json();
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}
