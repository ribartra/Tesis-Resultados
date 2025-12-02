/**
 * Audio Service para manejo de streaming de audio
 * Maneja la conversión y reproducción de audio desde el servidor RAG
 */

export class AudioService {
    constructor() {
        this.audioQueue = [];
        this.isPlayingAudio = false;
        this.audioChunkCount = 0;
    }

    /**
     * Procesa datos de audio desde el servidor
     * @param {string} audioData - Datos de audio (base64 o hex dependiendo del formato)
     * @param {string} format - Formato de los datos ('base64' o 'hex')
     * @returns {Audio|null} - Elemento de audio listo para reproducir
     */
    processAudioData(audioData, format = 'base64') {
        try {
            let audioBytes;
            
            if (format === 'base64') {
                // Procesar base64
                console.log(`[AudioService] Processing base64 audio data of length ${audioData.length}`);
                audioBytes = this.base64ToUint8Array(audioData);
            } else if (format === 'hex') {
                // Procesar hexadecimal
                console.log(`[AudioService] Processing hex audio data of length ${audioData.length}`);
                audioBytes = this.hexToUint8Array(audioData);
            } else {
                throw new Error(`Unsupported audio format: ${format}`);
            }

            // Crear blob de audio
            const audioBlob = new Blob([audioBytes], { type: 'audio/wav' });
            
            // Verificar que el blob tiene contenido
            if (audioBlob.size === 0) {
                console.warn('[AudioService] Generated empty audio blob');
                return null;
            }

            // Crear URL del blob
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Crear elemento de audio
            const audio = new Audio(audioUrl);
            audio.preload = 'auto';
            
            console.log(`[AudioService] Audio blob created successfully: ${audioBlob.size} bytes`);
            
            return audio;
        } catch (error) {
            console.error('[AudioService] Error processing audio data:', error);
            return null;
        }
    }

    /**
     * Convierte base64 a Uint8Array
     * @param {string} base64String - String en base64
     * @returns {Uint8Array} - Array de bytes
     */
    base64ToUint8Array(base64String) {
        try {
            // Decodificar base64
            const binaryString = atob(base64String);
            const bytes = new Uint8Array(binaryString.length);
            
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            return bytes;
        } catch (error) {
            console.error('[AudioService] Error converting base64 to bytes:', error);
            throw error;
        }
    }

    /**
     * Convierte hexadecimal a Uint8Array
     * @param {string} hexString - String hexadecimal
     * @returns {Uint8Array} - Array de bytes
     */
    hexToUint8Array(hexString) {
        try {
            // Verificar que es un string hex válido
            if (hexString.length % 2 !== 0) {
                throw new Error('Invalid hex string length');
            }
            
            const matches = hexString.match(/.{1,2}/g);
            if (!matches) {
                throw new Error('Invalid hex string format');
            }
            
            return new Uint8Array(matches.map(byte => parseInt(byte, 16)));
        } catch (error) {
            console.error('[AudioService] Error converting hex to bytes:', error);
            throw error;
        }
    }

    /**
     * Añade audio a la cola de reproducción
     * @param {Audio} audio - Elemento de audio
     * @param {Function} onAudio - Callback cuando se reproduce el audio
     */
    queueAudio(audio, onAudio = null) {
        if (!audio) {
            console.warn('[AudioService] Attempted to queue null audio');
            return;
        }

        audio.addEventListener('canplaythrough', () => {
            console.log(`[AudioService] Audio ready, adding to queue position ${this.audioQueue.length + 1}`);
            this.audioQueue.push({ audio, onAudio });
            this.playNextAudio();
        });

        audio.addEventListener('error', (e) => {
            console.error('[AudioService] Audio loading error:', e);
        });
    }

    /**
     * Reproduce el siguiente audio en la cola
     */
    playNextAudio() {
        if (this.audioQueue.length === 0 || this.isPlayingAudio) {
            return;
        }

        this.isPlayingAudio = true;
        const { audio, onAudio } = this.audioQueue.shift();
        const audioStartTime = performance.now();

        console.log(`[AudioService] Playing audio chunk ${++this.audioChunkCount}, ${this.audioQueue.length} remaining`);

        audio.addEventListener('ended', () => {
            const audioDuration = performance.now() - audioStartTime;
            console.log(`[AudioService] Audio chunk ${this.audioChunkCount} finished after ${audioDuration.toFixed(2)}ms`);
            this.isPlayingAudio = false;
            
            // Limpiar URL del blob para liberar memoria
            URL.revokeObjectURL(audio.src);
            
            this.playNextAudio();
        });

        audio.addEventListener('error', (e) => {
            console.error('[AudioService] Audio playback error:', e);
            this.isPlayingAudio = false;
            
            // Limpiar URL del blob
            URL.revokeObjectURL(audio.src);
            
            this.playNextAudio();
        });

        audio.play()
            .then(() => {
                if (onAudio) onAudio(audio);
            })
            .catch(error => {
                console.error('[AudioService] Error starting audio playback:', error);
                this.isPlayingAudio = false;
                URL.revokeObjectURL(audio.src);
                this.playNextAudio();
            });
    }

    /**
     * Limpia la cola de audio
     */
    clearQueue() {
        // Limpiar URLs de blob para liberar memoria
        this.audioQueue.forEach(({ audio }) => {
            URL.revokeObjectURL(audio.src);
        });
        
        this.audioQueue = [];
        this.isPlayingAudio = false;
        console.log('[AudioService] Audio queue cleared');
    }

    /**
     * Detecta automáticamente el formato de los datos de audio
     * @param {string} audioData - Datos de audio
     * @returns {string} - Formato detectado ('base64' o 'hex')
     */
    detectAudioFormat(audioData) {
        // Base64 usa caracteres A-Z, a-z, 0-9, +, /
        const base64Pattern = /^[A-Za-z0-9+/]*={0,2}$/;
        
        // Hex usa solo 0-9, A-F, a-f
        const hexPattern = /^[0-9A-Fa-f]+$/;
        
        if (base64Pattern.test(audioData)) {
            return 'base64';
        } else if (hexPattern.test(audioData)) {
            return 'hex';
        } else {
            console.warn('[AudioService] Unable to detect audio format, defaulting to base64');
            return 'base64';
        }
    }

    /**
     * Procesa y encola audio automáticamente detectando el formato
     * @param {string} audioData - Datos de audio
     * @param {Function} onAudio - Callback cuando se reproduce el audio
     */
    processAndQueueAudio(audioData, onAudio = null) {
        const format = this.detectAudioFormat(audioData);
        const audio = this.processAudioData(audioData, format);
        
        if (audio) {
            this.queueAudio(audio, onAudio);
        }
    }
}

// Crear instancia global del servicio de audio
export const audioService = new AudioService();