from fastapi import UploadFile, File
from openai import OpenAI
from sqlalchemy.orm import Session
from app.models.chat_thread import ChatThread
from app.models.chat_message import ChatMessage
from app.models.user import User
from typing import List
import logging
import time
import os
from dotenv import load_dotenv
from datetime import datetime
from textwrap import dedent
import random
from pydub import AudioSegment #pip install pydub 
from io import BytesIO
import re
import base64
import json
import asyncio
from app.whisper import transcribe
import uuid
import tempfile
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat_service")

# Especifica la ruta de FFmpeg manualmente si es necesario
AudioSegment.converter = ffmpeg_path = os.getenv("FFMPEG_PATH", "/usr/bin/ffmpeg")  # Usa el valor por defecto si no se encuentra

# Cargar variables de entorno desde el archivo .env
load_dotenv()

class ChatService:
    
    def __init__(self):
        self.timings = {}
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def generate_tempfile(self, audio_file: UploadFile = File(...)) -> str:
        logger.info(f"Received speech-to-text request: filename={audio_file.filename}, content_type={audio_file.content_type}")
        # Determinar la extensión basada en el content-type
        suffix = '.wav'
        if audio_file.content_type:
            if 'webm' in audio_file.content_type:
                suffix = '.webm'
            elif 'mp3' in audio_file.content_type:
                suffix = '.mp3'
            elif 'ogg' in audio_file.content_type:
                suffix = '.ogg'
        
        # Guardar el archivo de audio temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio_path = temp_audio.name
            content = await audio_file.read()
            temp_audio.write(content)
        
        logger.info(f"Audio saved temporarily at {temp_audio_path} ({len(content)} bytes)")
        return temp_audio_path
    
    def transcribe_openai(self, temp_file_path: str) -> str:
        try:
            # Usar la instancia de OpenAI del chat_service
            response = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=open(temp_file_path, "rb"),
                language="es",  # Especificar español
                prompt="The user prompt in Spanish audio to convert to text"
            )    
            transcription = response.text
            logger.info(f"Transcription successful: '{transcription[:50]}...' if len > 50")            
            # Limpiar archivo temporal
            os.unlink(temp_file_path)
            return transcription
        except Exception as e:
            logger.error(f"Error in transcribe_openai: {e}")
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e
    
    def transcribe_local(self, temp_file_path: str) -> str:
        try:# Llamar a la función de transcripción definida en whisper.py
            transcription = transcribe(temp_file_path)
            logger.info(f"Transcription successful: '{transcription[:50]}...' if len > 50")            
            # Limpiar archivo temporal
            os.unlink(temp_file_path)
            return transcription
        except Exception as e:
            logger.error(f"Error in transcribe_openai: {e}")
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise e

    def generate_voice(self, texto: str) -> bytes:
        """Generar voz desde texto usando TTS"""
        
        behavior = """
        Idioma: Español (Latinoamericano estándar).
        Género de voz: Femenina.
        Tono: Alegre, cálido y tranquilizador, proyectando cercanía y profesionalismo.
        Ritmo: Moderado y fluido, manteniendo claridad sin sonar apresurado.
        Pronunciación: Clara y precisa, con especial cuidado en términos médicos.
        Emoción: Positiva y empática, transmitiendo interés genuino por el bienestar del oyente.
        """
        
        # Si el texto está vacío, retornar bytes vacíos
        if not texto or texto.isspace():
            logger.warning("generate_voice called with empty text")
            return BytesIO().getvalue()

        logger.info(f"Generting voice ...")
        logger.info(f"*******************")
        try:
            logger.info(f"Generating voice for text of length: {len(texto)}")
            start_time = time.time() * 1000
            
            # Dividir el texto para mejorar latencia (frases completas)
            #chunks = re.split(r'[;,]\s+|(?<=[.!?])\s+', texto.strip())  
            #logger.info(f"Split text into {len(chunks)} chunks")
            
            #audio_final = AudioSegment.empty()

            # for i, chunk in enumerate(chunks):
            #     if not chunk or chunk.isspace():
            #         continue
                    
            #     chunk_start = time.time() * 1000
            #     logger.info(f"Generating audio for chunk {i+1}/{len(chunks)}: '{chunk}...' if len(chunk) > 30 else chunk")
                
            response = self.client.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="nova",
                input=texto,
                response_format="wav",  # Especificar formato explícitamente
                instructions=behavior,
                speed=4
            )
            
            # Leer el audio como wav (formato que devuelve OpenAI)
            #audio = AudioSegment.from_file(BytesIO(response.content), format="wav")
            #audio_final += audio
                
                # chunk_time = time.time() * 1000 - chunk_start
                # logger.info(f"Chunk {i+1} audio generated in {chunk_time:.2f}ms")

            # Exportar el audio final como wav (mismo formato que devuelve OpenAI)
            #buffer = BytesIO()
            #audio_final.export(buffer, format="wav")
            #audio.export(buffer, format="wav")
            total_time = time.time() * 1000 - start_time
            logger.info(f"Total voice generation completed in {total_time:.2f}ms")
            
            logger.info(f"*******************")
            logger.info(f"Preparing buffer for sending ...")
            #return buffer.getvalue()
            return BytesIO(response.content).getvalue()
        except Exception as e:
            logger.error(f"Error al generar audio: {str(e)}")
            # En caso de error, retornar bytes vacíos
            return BytesIO().getvalue()
            
    async def generate_voice_streaming(self, texto: str):
        """Generar voz desde texto usando TTS con streaming directo desde OpenAI"""        
        # Si el texto está vacío, retornar None
        if not texto or texto.isspace():
            logger.warning("generate_voice_streaming called with empty text")
            return None

        try:
            logger.info(f"Starting streaming TTS for text of length: {len(texto)}")
            start_time = time.time() * 1000
            
            # Implementar el streaming directo de OpenAI cuando esté disponible
            # Por ahora usamos la versión no streaming y simulamos chunks
            audio_bytes = self.generate_voice(texto)
            if audio_bytes:
                # Return the raw audio bytes directly without base64 encoding
                total_time = time.time() * 1000 - start_time
                logger.info(f"Streaming TTS completed in {total_time:.2f}ms")
                
                return audio_bytes
            return None
            
        except Exception as e:
            logger.error(f"Error en streaming TTS: {str(e)}")
            return None

    def get_system_message(self) -> dict:
        """Obtener el mensaje del sistema para el chatbot"""
        return {
            'role': 'system',
            'content': (
                dedent("""
                    ### **Propósito General**  
- **Qhali** asiste al usuario en la exploración de las diferentes áreas de interés y carreras que ofrece la PUCP.  
- Comienza solicitando la aceptación de términos y condiciones, luego muestra un texto de bienvenida, y finalmente guía al usuario a través de la consulta de carreras, áreas de interés y detalles específicos.  
- Cuando corresponda, Qhali consultará la información almacenada en el archivo XLSX (mediante file search) para presentar los textos de cada área y/o carrera solicitada por el usuario.
- Cuando se le consulte a Qhali sobre mencionar un conjunto de elementos como los listados de carreras y áreas, deberá solo listarlos y preguntarle al usuario sobre si quiere más información en estilo que mantenga la conversación por lo que Qhali deberá ser concisa y responder en estilo párrafo.
- Si Qhali no encuentra información en el archivo XLSX tendrá que decir "No hallé información sobre el tema en mi base de conocimientos". 
---

### **Estados Principales del Flujo**

1. **Pedido de Aceptación de Términos y Condiciones del Bot (PATCB)**  
   - **Función:** Es el punto de inicio. Qhali presenta los términos y condiciones y pide al usuario que los acepte para continuar.  
   - **Mensaje Ejemplo:**  
     > "¡Hola! Bienvenido(a) a la Oficina de Admisión PUCP. Soy Qhali y antes de empezar, necesito que leas y aceptes nuestros términos y condiciones para brindarte el mejor servicio de orientación. ¿Deseas aceptarlos?"  
   - **Transición Natural:**  
     - Si el usuario acepta (ej. “Sí, acepto”, “Estoy de acuerdo”), Qhali avanza al siguiente estado de MP.
     - Si el usuario se niega o no desea continuar, Qhali ofrece cerrar la conversación (estado Fin de Conversación) sin repreguntar.
     - Si el usuario quiere contactar a un agente Qhali ejecutará <mensaje-asesor-promesa-contacto-breve> y procederá a cerrar la conversación (FDC).

2. **Mensaje No Puedo Atención Medio (MNPAM)**  
   - **Función:** Estado de contingencia o fallback. Se utiliza cuando:
     - Qhali no puede satisfacer la solicitud en ese momento.
     - El usuario hace preguntas fuera de contexto que no corresponden a la información disponible.  
   - **Mensaje Ejemplo:**  
     > "Lo siento, en este momento no tengo la información que solicitas o no puedo procesar tu petición. ¿Deseas regresar al menú principal o prefieres cerrar la conversación?"  
   - **Transición Natural:**  
     - El usuario puede pedir volver al menú principal (estado de Texto Bienvenida u opciones) o finalizar la conversación.

3. **Texto Bienvenida / Menú Principal (MP)**  
   - **Función:** Qhali da la bienvenida una vez aceptados los términos y presenta las opciones principales.  
   - **Mensaje Ejemplo:**  
     > "¡Excelente! Has aceptado los términos y condiciones. Bienvenido(a) a Qhali, tu asistente de la Oficina de Admisión PUCP. Desde este menú, podrás conocer carreras disponibles, requisitos de admisión, o lo que necesites sobre la PUCP. ¿En qué puedo ayudarte hoy?"  
   - **Transición Natural:**  
     - Si el usuario quiere explorar carreras, Qhali lo conduce a **Información de Carreras (IC)** y únicamente lista las áreas sin explicarlas a detalle.
     - Si el usuario menciona directamente una carrera específica (por ejemplo, “Ingeniería Informática”), Qhali puede saltar directamente a la **Presentación de Carrera** correspondiente.
     - Si el usuario pregunta por requisitos, costos, becas o cualquier otro detalle, Qhali puede brindar información general o acceder a datos específicos (si están disponibles).  
     - Si el usuario quiere volver atrás o salir, se ofrece la posibilidad de cerrar la conversación (FDC).
     - Si el usuario pregunta algo que no se puede hallar en el archivo XLSX ni relacionar, Qhali procedera a comunicar la falta de información sobre el tema en su base de conocimiento (NFI).
     - Si el usuario quiere contactar a un agente Qhali ejecutará <mensaje-asesor-promesa-contacto-breve> y procederá a cerrar la conversación (FDC).

4. **Información de Carreras (IC)**  
   - **Función:** Qhali presenta un listado conciso de las áreas existentes, animando al usuario a explorar.  
   - **Mensaje Ejemplo:**  
     > "En la PUCP contamos con diversas áreas de interés, desde Artes hasta Ciencias e Ingeniería, pasando por Comunicaciones, Derecho y Empresa, Educación, Humanidades y muchas más. ¿Te interesa alguna de estas áreas en particular?"  
   - **Transición Natural:**  
     - Según la respuesta, Qhali deriva la conversación a la **Presentación de Área de Interés (PAIX)** apropiada.  
     - Si el usuario menciona directamente una carrera específica (por ejemplo, “Ingeniería Informática”), Qhali puede saltar directamente a la **Presentación de Carrera** correspondiente.
     - Si el usuario pregunta algo que no se puede hallar en el archivo XLSX ni relacionar, Qhali procedera a comunicar la falta de información sobre el tema en su base de conocimiento (NFI).
     - Si el usuario quiere contactar a un agente Qhali ejecutará <mensaje-asesor-promesa-contacto-breve> y procederá a cerrar la conversación (FDC).


5. **Presentación de Área de Interés X (PAIX)**  
   - **Función:** Cada “X” representa un área de interés (por ej., `<presentacion-areainteres-artes>`, `<presentacion-areainteres-cienciaseingenieria>`, etc.). Qhali recupera la descripción correspondiente del archivo XLSX (file search).  
   - **Mensaje Ejemplo:**  
     > "En el área de [X], encontrarás las siguientes carreras:"  
   - **Transición Natural:**  
     - El usuario puede solicitar ver las carreras específicas relacionadas a esa área, pasando a **Presentación de Carrera** (por ejemplo, `<presentacion-carrera-arquitectura>`).  
     - Si el usuario quiere regresar al menú principal o cambiar a otra área, Qhali permite hacerlo sin problema.
     - Si el usuario pregunta algo que no se puede hallar en el archivo XLSX ni relacionar, Qhali procedera a comunicar la falta de información sobre el tema en su base de conocimiento (NFI).
     - Si el usuario quiere contactar a un agente Qhali ejecutará <mensaje-asesor-promesa-contacto-breve> y procederá a cerrar la conversación (FDC).


6. **Presentación de Carrera Específica**  
   - **Función:** Muestra la información detallada de la carrera solicitada. Está almacenada en el archivo XLSX con etiquetas como `<presentacion-carrera-derecho>`, `<presentacion-carrera-industrial>`, etc.  
   - **Mensaje Ejemplo:**  
     > "Aquí tienes la información de [nombre de la carrera] que solicitaste: perfil profesional y otros datos relevantes."  
   - **Transición Natural:**  
     - El usuario puede solicitar más detalles, preguntar por otra carrera, volver al menú principal o incluso cerrar la conversación.
     - Si el usuario pregunta algo que no se puede hallar en el archivo XLSX ni relacionar, Qhali procedera a comunicar la falta de información sobre el tema en su base de conocimiento (NFI).
     - Si el usuario quiere contactar a un agente Qhali ejecutará <mensaje-asesor-promesa-contacto-breve> y procederá a cerrar la conversación (FDC).


7. **No found information (NFI)**
   - **Función:** Comunica al usuario que no encuentra el tema que consultó.  
   - **Mensaje Ejemplo:**  
     > "Lo siento, pero no hallé información sobre el tema en mi base de conocimientos".
   - **Transición Natural:**
     - Si el usuario quiere contactar a un agente Qhali ejecutará <mensaje-asesor-promesa-contacto-breve> y procederá a cerrar la conversación (FDC).


8. **Fin de Conversación (FDC)**  
   - **Función:** Cierre amigable de la interacción.  
   - **Mensaje Ejemplo:**  
     > "Muchas gracias por utilizar nuestros servicios de orientación. ¡Te deseamos lo mejor en tu búsqueda vocacional! Si tienes más consultas en el futuro, aquí estaré. ¡Hasta pronto!"

---

### **Reglas de Interacción y Comportamiento de Qhali**

1. **Interpretación de Lenguaje Natural:**  
   - Qhali debe aceptar frases como “Sí, me interesa artes”, “Quiero saber sobre Ingeniería Industrial” o “no deseo seguir” y derivar la conversación de manera coherente al estado correspondiente.  
   - No se requiere que el usuario mencione explícitamente las siglas (PATCB, MNPAM, etc.); se deduce la intención a partir de la respuesta.

2. **Consulta de Variables en el Archivo XLSX (file search):**  
   - Para presentar contenido de áreas de interés o carreras específicas, Qhali debe buscar la etiqueta en el archivo XLSX y mostrar el contenido almacenado.  
   - Este proceso permite mantener la información centralizada y actualizada, sin necesidad de modificar el prompt.
   - Si la búsqueda por file search no brinda resultados Qhali debe comunicar "No pude encontrar información en mi base de conocimientos"

3. **Fallback y Manejo de Errores (MNPAM):**  
   - Si en cualquier estado el usuario formula peticiones fuera del alcance de Qhali o si no se dispone de la información, se procede a MNPAM con un mensaje cordial de disculpa y se ofrece volver al menú principal o cerrar la conversación.

4. **Flexibilidad en la Navegación:**  
   - El usuario puede cambiar de área de interés, preguntar por otra carrera o volver al menú principal en cualquier momento. Qhali debe adaptarse sin forzar un orden lineal estricto.  
   - Si el usuario desea terminar la conversación en cualquier momento, Qhali ofrece una despedida adecuada.
   - Si el usuario quiere contactar a un agente, Qhali simulará y comunicará al usuario que se van a contactar más rápido posible con un agente humano Qhali procederá a cerrar la conversación (FDC).

5. **Tono y Estilo de Respuesta:**  
   - Empático, claro y didáctico. Qhali representa a la Oficina de Admisión PUCP, por lo que debe mantener un lenguaje formal pero cercano, transmitiendo confianza y profesionalismo.

---

### **Ejemplo de Secuencia Conversacional**

1. **Qhali (Estado PATCB):**  
   > "Bienvenido(a) a la Oficina de Admisión PUCP. Antes de comenzar, te invito a aceptar nuestros términos y condiciones. ¿Te parece bien?"
2. **Usuario:** "Acepto."  
3. **Qhali (Menú Principal MP):**  
   > "¡Perfecto! Soy Qhali, tu asistente de orientación. ¿Te gustaría conocer nuestras áreas de interés, consultar una carrera específica o hablar sobre requisitos de admisión?"
4. **Usuario:** "Me interesa algo de ciencias, ¿qué carreras tienen?"  
5. **Qhali (IC → PAIX Ciencias e Ingeniería):**  
   > "Contamos con diversas carreras de ciencias e ingeniería. Permíteme consultarlas…"  
   (Qhali hace *file search* para `<presentacion-areainteres-cienciaseingenieria>` y muestra el contenido.)  
6. **Usuario:** "Ingeniería Informática me llama la atención."  
7. **Qhali (Presentación de Carrera Específica):**  
   > "Aquí tienes los detalles de Ingeniería Informática…"  
   (Qhali hace *file search* para `<presentacion-carrera-informática>` y muestra el contenido.)  
8. **Usuario:** "Suena interesante. Gracias, quisiera salir por ahora."  
9. **Qhali (FDC):**  
   > "¡Muchas gracias por tu interés! Cuando gustes, estoy a tu disposición para más consultas. ¡Hasta pronto!"

                     """
                )
            )
        }
    
    def create_thread(self, db: Session, user_id: int = None, category: str = "qhali-llama", title: str = "Conversación" + datetime.now().strftime('%d-%m-%y %H:%M') ):
        """Crear un nuevo hilo en la base de datos"""
        logger.info(f"Creating new thread with title: {title}")
        new_thread = ChatThread(
            id_user=user_id,
            category=category,
            title=title
        )
        db.add(new_thread)
        db.commit()
        db.refresh(new_thread)
        logger.info(f"Thread created with ID: {new_thread.id_thread}")
        return new_thread

    def send_message_to_openai(self, messages: List[dict]) -> str:
        """Enviar mensaje a OpenAI y recibir la respuesta"""
        try:
            # Asegurarse de que los mensajes sean una lista de objetos con las claves "role" y "content"
            if not isinstance(messages, list):
                raise ValueError("Los mensajes deben estar en formato de lista.")

            logger.info(f"Sending message to OpenAI with {len(messages)} messages")
            start_time = time.time() * 1000
            
            stream = self.client.responses.create(
                model="gpt-4-turbo-preview",
                input=messages,
                temperature=0.1,
                top_p=0.2,
                store=False,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": ["vs_68198578f8cc81918938e54fe53c32c1"]
                }],
                stream=True
            )
            
            assistant_message = response.output_text
            
            elapsed_time = time.time() * 1000 - start_time
            logger.info(f"Received OpenAI response in {elapsed_time:.2f}ms")
            
            return assistant_message
        except Exception as e:
            logger.error(f"Error al enviar mensaje a OpenAI: {str(e)}")
            raise Exception("Error al procesar el mensaje con OpenAI.")
            
    def send_message_to_openai_stream(self, messages: List[dict]):
        """Enviar mensaje a OpenAI y recibir la respuesta en streaming"""
        try:
            # Asegurarse de que los mensajes sean una lista de objetos con las claves "role" y "content"
            if not isinstance(messages, list):
                raise ValueError("Los mensajes deben estar en formato de lista.")

            logger.info(f"Starting streaming request to OpenAI with {len(messages)} messages")
            start_time = time.time() * 1000
            
            # Usando el nuevo cliente OpenAI con streaming
            stream = self.client.responses.create(
                model="gpt-4-turbo-preview",
                input=messages,
                temperature=0.1,
                top_p=0.2,
                store=False,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": ["vs_68198578f8cc81918938e54fe53c32c1"]
                }],
                stream=True
            )
            
            elapsed_time = time.time() * 1000 - start_time
            logger.info(f"OpenAI stream connection established in {elapsed_time:.2f}ms")
            
            return stream
        except Exception as e:
            logger.error(f"Error al enviar mensaje a OpenAI en streaming: {str(e)}")
            raise Exception("Error al procesar el mensaje con OpenAI en streaming.")
            
    async def send_message_to_thread_text_stream(self, db: Session, thread_id: int, user_message: str):
        """Enviar mensaje de texto a un hilo con streaming (sin audio)"""
        # Registro de tiempo para métricas
        self.timings['start_request'] = time.time() * 1000
        logger.info(f"Starting text-only stream for thread {thread_id}")
        
        # Guardar el mensaje del usuario en la base de datos
        user_message_record = ChatMessage(
            thread_id=thread_id,
            user_id=None,  # Cambiar por user_id si está disponible
            message=user_message,
            role="user"
        )
        db.add(user_message_record)
        db.commit()
        logger.info(f"User message saved to database for thread {thread_id}")
        
        # Recuperar el historial de mensajes para enviarlos a OpenAI (últimos 15 mensajes)
        messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.created_at.desc()).limit(15).all()
        logger.info(f"Retrieved {len(messages)} messages from thread history")
        
        # Formatear mensajes para OpenAI
        formatted_messages = [self.get_system_message()] + [
            {'role': msg.role, 'content': msg.message} for msg in reversed(messages)
        ]
        
        # Obtener el stream de respuestas
        stream = self.send_message_to_openai_stream(formatted_messages)
        logger.info("Started OpenAI stream")
        
        # Variables para acumular la respuesta completa
        complete_response = ""
        token_count = 0
        
        # Procesar los eventos del stream
        for event in stream:
            if event.type == "response.output_text.delta":
                delta_text = event.delta
                if not all(c in ['#', '*', '-'] for c in delta_text.strip()):
                    complete_response += delta_text
                    token_count += 1
                
                if token_count % 10 == 0:
                    logger.info(f"Processed {token_count} tokens so far")
                
                # Enviar inmediatamente el delta de texto con formato SSE
                yield f"data: {json.dumps({'type': 'text_delta', 'text': delta_text})}\n\n"
                # Enviar un comentario vacío para forzar el flush
                yield f": keep-alive\n\n"

            elif event.type == "response.completed":
                # Registro de tiempos
                self.timings['end_request'] = time.time() * 1000
                self.timings['total_request_time'] = self.timings['end_request'] - self.timings['start_request']
                
                logger.info(f"Stream of thread text stream completed in {self.timings['total_request_time']:.2f}ms, generated {token_count} tokens")
                yield f"data: {json.dumps({'type': 'completion', 'thread_id': thread_id, 'timing': self.timings})}\n\n"

        # Guardar la respuesta completa en la base de datos
        assistant_message_record = ChatMessage(
            thread_id=thread_id,
            message=complete_response,
            role="assistant"
        )
        db.add(assistant_message_record)
        db.commit()
        logger.info(f"Assistant response saved to database for thread {thread_id}")

    async def send_message_to_thread_stream(self, db: Session, thread_id: int, user_message: str):
        """Enviar mensaje a un hilo existente, procesarlo y obtener respuesta de OpenAI en streaming con audio"""
        
        # Registro de tiempo para métricas
        self.timings['start_request'] = time.time() * 1000
        logger.info(f"Starting audio+text stream for thread {thread_id}")

        yield f"data: {json.dumps({'type': 'text_delta', 'text': ''})}\n\n"
        yield f": keep-alive\n\n"  # Force a flush

        # Guardar el mensaje del usuario en la base de datos
        user_message_record = ChatMessage(
            thread_id=thread_id,
            user_id=None,  # Cambiar por user_id si está disponible
            message=user_message,
            role="user"
        )
        db.add(user_message_record)
        db.commit()
        logger.info(f"User message saved to database for thread {thread_id}")
        
        try:
            # Recuperar el historial de mensajes para enviarlos a OpenAI (últimos 15 mensajes)
            messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.created_at.desc()).limit(15).all()
            logger.info(f"Retrieved {len(messages)} messages from thread history")
            
            # # Formatear mensajes para OpenAI
            formatted_messages = [self.get_system_message()] + [
                {'role': msg.role, 'content': msg.message} for msg in reversed(messages)
            ]
            # formatted_messages = [
            #     {'role': 'user', 'content': user_message} 
            # ]
            
            # Obtener el stream de respuestas
            stream = self.send_message_to_openai_stream(formatted_messages)
            # logger.info("Started OpenAI send message thread stream")
                        
            # # Variables para acumular la respuesta completa y el buffer de texto
            complete_response = ""
            text_buffer = ""
            # sentence_pattern = r'(?<=[.!?])\s+'
            sentence_pattern = r'(?<=[.!?]|[,!?])\s+'
            token_count = 0
            audio_chunks = 0
            
            # # Cola asíncrona para generar audio sin bloquear los deltas de texto
            audio_generation_queue = []
            has_posted = False

            # if not has_posted:                            
            #     if self.es_rechazo_terminos(user_message):
            #         await self.fire_and_forget_post("hn1-esp")
            #     elif self.es_fin_conversacion(user_message) or self.es_pedido_agente(user_message) or self.es_saludo(user_message):
            #         await self.fire_and_forget_post("hn3-esp")
            #     else:
            #         action = random.choice(["hn4", "hn5","hn6","hn7","hn8",])
            #         await self.fire_and_forget_post(f"{action}-esp")
            #     has_posted= True

                    
            # Procesar los eventos del stream
            for event in stream:
                # Si el evento es de tipo delta de texto, procesarlo
                if event.type != "response.output_text.delta":
                    logger.info(f"Processing event : {event.type}")
                if event.type == "response.output_text.delta":
                    delta_text = event.delta

                    # Filtrar texto que solo contiene caracteres decorativos
                    if not all(c in ['#', '*', '-'] for c in delta_text.strip()):
                        complete_response += delta_text
                        text_buffer += delta_text
                        # token_count += 1
                    
            #         # if token_count % 10 == 0:
            #         #     logger.info(f"Processed {token_count} tokens so far")
                    
            #         # Enviar delta de texto inmediatamente
                    logger.info(delta_text)
                    yield f"data: {json.dumps({'type': 'text_delta', 'text': delta_text})}\n\n"
            #         # Enviar un comentario vacío para forzar el flush
                    yield f": keep-alive\n\n"
                    await asyncio.sleep(0.01)
            #         # Verificar si tenemos oraciones completas para generar audio
                    if re.search(sentence_pattern, text_buffer):
                        sentences = re.split(sentence_pattern, text_buffer)
                        logger.info(f"Found complete sentence(s): '{sentences}...'")
                        # Si hay más de una oración, todas excepto la última están completas
                        if len(sentences) > 1:
                            completed_sentences = sentences[:-1]
                            complete_text = ' '.join(completed_sentences)
                            
                            logger.info(f"Found complete sentence(s) ({len(completed_sentences)}): '{complete_text}'")
                            audio_chunks += 1

                            # Tarea asíncrona para generar audio
                            async def generate_audio(text):
                                try:
                                    logger.info(f"Generating audio for text: '{text}...'")
                                    audio_start_time = time.time() * 1000
                                    
                                    logger.info(f"+++++++++++++++++ START generating voice ++++++++")
                                    # Usar el método de streaming si está disponible
                                    audio_bytes = await self.generate_voice_streaming(text)
                                    
                                    audio_time = time.time() * 1000 - audio_start_time
                                    logger.info(f"Audio generated in {audio_time:.2f}ms")
                                    logger.info(f"+++++++++++++++++ END generating voice ++++++++")
                                    return audio_bytes
                                except Exception as e:
                                    logger.error(f"Error en generación de audio: {str(e)}")
                                    return None
                            
                            # Agregar tarea a la cola
                            # audio_generation_queue.append(asyncio.create_task(generate_audio(complete_text)))
                            audio_bytes = await generate_audio(complete_text)
                            if audio_bytes:
                                    audio_hex = audio_bytes.hex()
                                    yield f"data: {json.dumps({'type': 'audio_bytes', 'text': '', 'audio_hex': audio_hex})}\n\n"
                            # Actualizar el buffer para mantener solo la oración incompleta
                            text_buffer = sentences[-1]
                            

                # Si es el evento de finalización, procesar cualquier texto restante
                elif event.type == "response.completed":
                    logger.info("Received completion event from OpenAI")

                    # Si queda texto en el buffer, generar audio para él
                    if text_buffer:
                        logger.info(f"Processing final text buffer: '{text_buffer[:30]}...'")
                        audio_chunks += 1
                        
                        async def generate_final_audio(text):
                            try:
                                logger.info(f"Generating final audio for text: '{text[:30]}...'")
                                audio_start_time = time.time() * 1000
                                
                                # Usar el método de streaming si está disponible
                                audio_bytes = await self.generate_voice_streaming(text)
                                
                                audio_time = time.time() * 1000 - audio_start_time
                                logger.info(f"Final audio generated in {audio_time:.2f}ms")
                                
                                return audio_bytes
                            except Exception as e:
                                logger.error(f"Error en generación de audio final: {str(e)}")
                                return None
                        
                        audio_generation_queue.append(asyncio.create_task(generate_final_audio(text_buffer)))



            #         # Registro de tiempos de la respuesta
            #         self.timings['end_request'] = time.time() * 1000
            #         self.timings['total_request_time'] = self.timings['end_request'] - self.timings['start_request']
            
            # # Esperar a todas las tareas de generación de audio y enviar cuando estén listas
            if audio_generation_queue:
                logger.info(f"Processing {len(audio_generation_queue)} audio generation tasks")
                for i, audio_task in enumerate(audio_generation_queue):
                    try:
                        audio_bytes = await audio_task
                        if audio_bytes:
                            logger.info(f"------------ Sending audio chunk {i+1}/{len(audio_generation_queue)}")
                            # Enviar evento de audio con bytes codificados en hex para el SSE
                            audio_hex = audio_bytes.hex()
                            yield f"data: {json.dumps({'type': 'audio_bytes', 'text': '', 'audio_hex': audio_hex})}\n\n"
                    except Exception as e:
                        logger.error(f"Error al procesar tarea de audio {i}: {str(e)}")
            #await self.fire_and_forget_post("cero")
                    
            # Enviar evento de finalización
            # logger.info(f"Stream thread message completed in {self.timings['total_request_time']:.2f}ms, generated {token_count} tokens and {audio_chunks} audio chunks")
            yield f"data: {json.dumps({'type': 'completion', 'thread_id': thread_id, 'timing': self.timings})}\n\n"
            
            # Guardar la respuesta completa en la base de datos
            assistant_message_record = ChatMessage(
                thread_id=thread_id,
                message=complete_response,
                role="assistant"
            )
            db.add(assistant_message_record)
            db.commit()
            logger.info(f"Assistant response saved to database for thread {thread_id}")
        
        except Exception as e:
            logger.error(f"Error en stream de mensajes: {str(e)}")
            # Informar del error al cliente
            yield f"data: {json.dumps({'type': 'error', 'message': 'Error al procesar el mensaje'})}\n\n"
            # Registrar el error también en la BD para futuras consultas
            try:
                error_message = ChatMessage(
                    thread_id=thread_id,
                    message="Error al procesar el mensaje: " + str(e),
                    role="system"
                )
                db.add(error_message)
                db.commit()
            except:
                pass  # Evitar errores en cascada al intentar registrar el error

    async def create_and_run_thread_stream(self, db: Session, user_message: str, user_id: int = None):
        """Crear un nuevo hilo y ejecutar el primer mensaje con OpenAI en streaming"""
        
        # Registro de tiempo para métricas
        self.timings['start_request'] = time.time() * 1000
        logger.info("Starting new thread creation with streaming")
        
        # Obtener el hilo de conversación con el código de la conversación
        thread_title = f"Conversación {datetime.now().strftime('%d-%m-%y %H:%M')}"
        
        # Crear el hilo de conversación
        thread = self.create_thread(db, user_id, title=thread_title)
        
        # Verificación de si el hilo fue creado correctamente
        if not thread or not hasattr(thread, 'id_thread'):
            logger.error("Error creating thread: No valid id_thread generated")
            raise Exception("Error al crear el hilo. No se generó un id_thread válido.")
                    
        # Yield la info del hilo creado primero
        logger.info(f"Thread created with ID: {thread.id_thread}, sending thread info to client")
        yield f"data: {json.dumps({'type': 'thread_created', 'id_thread': thread.id_thread, 'title': thread.title})}\n\n"
        # Forzar flush con un comentario vacío
        yield f": keep-alive\n\n"
        
        # Guardar el mensaje del usuario en la base de datos
        user_message_record = ChatMessage(
            thread_id=thread.id_thread,
            user_id=user_id,
            message=user_message,
            role="user"
        )
        db.add(user_message_record)
        db.commit()
        logger.info(f"User message saved to database for new thread {thread.id_thread}")
        
        try:
            # Formatear mensajes para OpenAI
            formatted_messages = [self.get_system_message()] + [
                {'role': 'user', 'content': user_message}  # Solo el primer mensaje del usuario
            ]
            
            # Obtener el stream de respuestas
            stream = self.send_message_to_openai_stream(formatted_messages)
            logger.info("Started OpenAI stream")

            # await self.fire_and_forget_post("hn3-esp")
            
            # Variables para acumular la respuesta completa y el buffer de texto
            complete_response = ""
            text_buffer = ""
            sentence_pattern = r'(?<=[.!?])\s+'
            token_count = 0
            audio_chunks = 0
            has_posted = False
            # if not has_posted:                            
            #     if self.es_rechazo_terminos(user_message):
            #         await self.fire_and_forget_post("hn1-esp")
            #     elif self.es_fin_conversacion(user_message) or self.es_pedido_agente(user_message) or self.es_saludo(user_message):
            #         await self.fire_and_forget_post("hn3-esp")
            #     else:
            #         action = random.choice(["hn4", "hn5","hn6","hn7","hn8",])
            #         await self.fire_and_forget_post(f"{action}-esp")
            #     has_posted= True
            
            # Cola asíncrona para generar audio sin bloquear los deltas de texto
            audio_generation_queue = []
                    
            # Procesar los eventos del stream
            for event in stream:
                # Si el evento es de tipo delta de texto, procesarlo
                if event.type == "response.output_text.delta":
                    delta_text = event.delta
                    # Filtrar texto que solo contiene caracteres decorativos
                    if not all(c in ['#', '*', '-'] for c in delta_text.strip()):
                        complete_response += delta_text
                        text_buffer += delta_text
                        token_count += 1
                    
                    if token_count % 10 == 0:
                        logger.info(f"Processed {token_count} tokens so far")
                    
                    # Enviar delta de texto inmediatamente
                    yield f"data: {json.dumps({'type': 'text_delta', 'text': delta_text})}\n\n"
                    # Enviar un comentario vacío para forzar el flush
                    yield f": keep-alive\n\n"
                    
                    # Verificar si tenemos oraciones completas para generar audio
                    if re.search(sentence_pattern, text_buffer):
                        sentences = re.split(sentence_pattern, text_buffer)
                        
                        # Si hay más de una oración, todas excepto la última están completas
                        if len(sentences) > 1:
                            completed_sentences = sentences[:-1]
                            complete_text = ' '.join(completed_sentences)
                            
                            logger.info(f"Found complete sentence(s) ({len(completed_sentences)}): '{complete_text[:30]}...'")
                            audio_chunks += 1
                            
                            # Tarea asíncrona para generar audio
                            async def generate_audio(text):
                                try:
                                    logger.info(f"Generating audio for text: '{text[:30]}...'")
                                    audio_start_time = time.time() * 1000
                                    
                                    # Usar el método de streaming si está disponible
                                    audio_bytes = await self.generate_voice_streaming(text)
                                    
                                    audio_time = time.time() * 1000 - audio_start_time
                                    logger.info(f"Audio generated in {audio_time:.2f}ms")
                                    
                                    return audio_bytes
                                except Exception as e:
                                    logger.error(f"Error en generación de audio: {str(e)}")
                                    return None
                            
                            # Agregar tarea a la cola
                            audio_generation_queue.append(asyncio.create_task(generate_audio(complete_text)))

                            # Actualizar el buffer para mantener solo la oración incompleta
                            text_buffer = sentences[-1]
                
                # Si es el evento de finalización, procesar cualquier texto restante
                elif event.type == "response.completed":
                    logger.info("Received completion event from OpenAI")
                    
                    # Si queda texto en el buffer, generar audio para él
                    if text_buffer:
                        logger.info(f"Processing final text buffer: '{text_buffer[:30]}...'")
                        audio_chunks += 1
                        
                        async def generate_final_audio(text):
                            try:
                                logger.info(f"Generating final audio for text: '{text[:30]}...'")
                                audio_start_time = time.time() * 1000
                                
                                # Usar el método de streaming si está disponible
                                audio_bytes = await self.generate_voice_streaming(text)
                                
                                audio_time = time.time() * 1000 - audio_start_time
                                logger.info(f"Final audio generated in {audio_time:.2f}ms")
                                
                                return audio_bytes
                            except Exception as e:
                                logger.error(f"Error en generación de audio final: {str(e)}")
                                return None
                        
                        audio_generation_queue.append(asyncio.create_task(generate_final_audio(text_buffer)))
                    
                    # Registro de tiempos de la respuesta
                    self.timings['end_request'] = time.time() * 1000
                    self.timings['total_request_time'] = self.timings['end_request'] - self.timings['start_request']
            
            # Esperar a todas las tareas de generación de audio y enviar cuando estén listas
            if audio_generation_queue:
                logger.info(f"Processing {len(audio_generation_queue)} audio generation tasks")
                for i, audio_task in enumerate(audio_generation_queue):
                    try:
                        audio_bytes = await audio_task
                        if audio_bytes:
                            logger.info(f"Sending audio chunk {i+1}/{len(audio_generation_queue)}")
                            # Enviar evento de audio con bytes codificados en hex para el SSE
                            audio_hex = audio_bytes.hex()
                            yield f"data: {json.dumps({'type': 'audio_bytes', 'text': '', 'audio_hex': audio_hex})}\n\n"
                    except Exception as e:
                        logger.error(f"Error al procesar tarea de audio {i}: {str(e)}")
            #await self.fire_and_forget_post("cero")
            
            # Enviar evento de finalización
            logger.info(f"Stream created and run completed in {self.timings['total_request_time']:.2f}ms, generated {token_count} tokens and {audio_chunks} audio chunks")
            yield f"data: {json.dumps({'type': 'completion', 'thread_id': thread.id_thread, 'timing': self.timings})}\n\n"
            
            # Guardar la respuesta completa en la base de datos
            assistant_message_record = ChatMessage(
                thread_id=thread.id_thread,
                message=complete_response,
                role="assistant"
            )
            db.add(assistant_message_record)
            db.commit()
            logger.info(f"Assistant response saved to database for thread {thread.id_thread}")
        
        except Exception as e:
            logger.error(f"Error en stream de mensajes: {str(e)}")
            # Informar del error al cliente
            yield f"data: {json.dumps({'type': 'error', 'message': 'Error al procesar el mensaje'})}\n\n"
            # Registrar el error también en la BD para futuras consultas
            try:
                error_message = ChatMessage(
                    thread_id=thread.id_thread,
                    message="Error al procesar el mensaje: " + str(e),
                    role="system"
                )
                db.add(error_message)
                db.commit()
            except Exception as inner_e:
                logger.error(f"Error al guardar el mensaje de error: {str(inner_e)}")
                pass  # Evitar errores en cascada al intentar registrar el error

    def create_and_run_thread(self, db: Session, user_message: str, user_id: int = None) -> dict:
        """Crear un nuevo hilo y ejecutar el primer mensaje con OpenAI"""

        # Registro de tiempo para métricas
        self.timings['start_request'] = time.time() * 1000

        # Obtener el hilo de conversación con el código de la conversación
        thread_title = f"Conversación {datetime.now().strftime('%d-%m-%y %H:%M')}"

        # Crear el hilo de conversación
        thread = self.create_thread(db, user_id, title=thread_title)

        # Verificación de si el hilo fue creado correctamente
        if not thread or not hasattr(thread, 'id_thread'):
            raise Exception("Error al crear el hilo. No se generó un id_thread válido.")

        # Crear mensaje del usuario en el hilo
        user_message_record = ChatMessage(
            thread_id=thread.id_thread,
            user_id=user_id,
            message=user_message,
            role="user"
        )
        db.add(user_message_record)
        db.commit()

        # Formatear el primer mensaje y el mensaje del sistema para OpenAI
        formatted_messages = [self.get_system_message()] + [
            {'role': 'user', 'content': user_message}  # Solo el primer mensaje del usuario
        ]

        # Enviar mensaje a OpenAI y obtener la respuesta del asistente
        assistant_message = self.send_message_to_openai(formatted_messages)

        # Guardar la respuesta del asistente en la base de datos
        assistant_message_record = ChatMessage(
            thread_id=thread.id_thread,
            message=assistant_message,
            role="assistant"
        )
        db.add(assistant_message_record)
        db.commit()

        # Registro de tiempos de la respuesta
        self.timings['end_request'] = time.time() * 1000
        self.timings['total_request_time'] = self.timings['end_request'] - self.timings['start_request']

        # Retornar la respuesta junto con los tiempos de la solicitud
        return {
            'id_thread': thread.id_thread,  # Devolviendo el thread_id generado
            'title': thread.title,  # Devolviendo el título del hilo
            'message': assistant_message,  # Mensaje del asistente
            'timing': self.timings  # Tiempos de la solicitud
        }
    
    def get_thread_history(self, db: Session, thread_id: int) -> List[dict]:
        """Obtener el historial de mensajes de un hilo"""
        try:
            messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.created_at.asc()).all()
            return [{"role": msg.role, "message": msg.message} for msg in messages]
        except Exception as e:
            logging.error(f"Error al obtener el historial del hilo {thread_id}: {str(e)}")
            raise Exception(f"Error al cargar el historial del hilo {thread_id}.")

    def send_message_to_thread(self, db: Session, thread_id: int, user_message: str) -> dict:
        """Enviar mensaje a un hilo existente, procesarlo y obtener respuesta de OpenAI"""

        # Registro de tiempo para métricas
        self.timings['start_request'] = time.time() * 1000

        # Guardar el mensaje del usuario en la base de datos
        user_message_record = ChatMessage(
            thread_id=thread_id,
            user_id=None,  # Cambiar por user_id si está disponible
            message=user_message,
            role="user"
        )
        db.add(user_message_record)
        db.commit()

        # Recuperar el historial de mensajes para enviarlos a OpenAI (últimos 15 mensajes)
        messages = db.query(ChatMessage).filter(ChatMessage.thread_id == thread_id).order_by(ChatMessage.created_at.desc()).limit(15).all()

        # Formatear mensajes para OpenAI
        formatted_messages = [self.get_system_message()] + [
            {'role': msg.role, 'content': msg.message} for msg in reversed(messages)
        ]

        # Enviar mensaje a OpenAI y obtener la respuesta del asistente
        assistant_message = self.send_message_to_openai(formatted_messages)

        # Guardar la respuesta del asistente en la base de datos
        assistant_message_record = ChatMessage(
            thread_id=thread_id,
            message=assistant_message,
            role="assistant"
        )
        db.add(assistant_message_record)
        db.commit()

        # Registro de tiempos de la respuesta
        self.timings['end_request'] = time.time() * 1000
        self.timings['total_request_time'] = self.timings['end_request'] - self.timings['start_request']

        # Retornar el mensaje del asistente y el hilo
        return {
            'message': assistant_message,
            'timing': self.timings  # Tiempos de la solicitud
        }
    
    def save_thread_history(self, db: Session, thread_id: int, messages: List[dict]) -> dict:
        """Guardar el historial de mensajes de un hilo"""
        try:
            for msg in messages:
                user_id = msg.get('user_id') if msg['role'] == 'user' else None
                message_record = ChatMessage(
                    thread_id=thread_id,
                    user_id=user_id,
                    message=msg['message'],
                    role=msg['role']
                )
                db.add(message_record)
            db.commit()
            return {"success": True, "message": "Historial guardado"}
        except Exception as e:
            logging.error(f"Error al guardar el historial del hilo {thread_id}: {str(e)}")
            raise Exception(f"Error al guardar el historial del hilo {thread_id}.")

    def es_rechazo_terminos(self, user_input: str) -> bool:
        texto = user_input.lower()

        patrones_rechazo = [
            r'\bno\s+(quiero|deseo|acepto|aceptaré|continuar|seguir)\b',
            r'\bno\s+voy\s+a\s+(aceptar|continuar|seguir)\b',
            r'\bme\s+niego(\s+a\s+(aceptar|continuar|seguir))?\b',
            r'\bno\s+(quiero|pienso|deseo)\b.*\b(términos?|condiciones?)\b',
            r'\bno\b.*\bacepto\b',
            r'\brechazo\b',
            r'\bdeclino\b',
            r'\bme\s+rehuso\b',
            r'\bno\s+acept[oé]?\b',  # para aceptar variantes tipo "acepto", "acepté"
            r'\bni\s+acepto\b',
            r'\bni\s+quiero\b',
        ]

        return any(re.search(p, texto) for p in patrones_rechazo)
    
    def es_fin_conversacion(self,user_input: str) -> bool:
        texto = user_input.lower()
        patrones = [
            r'\b(gracias|muchas gracias|okey|ya está|todo bien)\b.*\b(adiós|chau|terminar|cerrar|eso es todo|eso sería todo|hasta luego)\b',
            r'\bterminar\b.*\bconversación\b',
            r'\bcerrar\b.*\bchat\b',
            r'\bye\b|\badiós\b|\bhasta luego\b',
            r'\bno\b.*\b(tengo|necesito|más preguntas|más dudas)\b',
            r'\bterminé\b|\bya fue\b|\bya acabé\b'
        ]
        return any(re.search(p, texto) for p in patrones)
    
    def es_pedido_agente(self,user_input: str) -> bool:
        texto = user_input.lower()
        patrones = [
            r'\bquiero\b.*\bagente\b',
            r'\bpuedo\b.*\bhablar\b.*\bhumano\b',
            r'\bnecesito\b.*\bpersona\b',
            r'\bquiero\b.*\bcontactar\b.*\balguien\b',
            r'\bme\b.*\bderiven?\b.*\bhumano\b',
            r'\bchat\b.*\bagente\b',
            r'\bpuedo\b.*\bcomunicarme\b.*\b(una persona|un agente|humano)\b',
        ]
        return any(re.search(p, texto) for p in patrones)
    
    def es_saludo(self, user_input: str) -> bool:
        texto = user_input.lower()

        patrones = [
            r'\bhola\b',
            r'\bholi(s)?\b',
            r'\bholita(s)?\b',
            r'\bhey\b',
            r'\bhello\b',
            r'\bhi\b',
            r'\bsaludos\b',
            r'\bqué tal\b',
            r'\bcomo (estás|estais|anda[s]?)\b',
            r'\bbuenos días\b',
            r'\bbuenas tardes\b',
            r'\bbuenas noches\b',
            r'\bqué onda\b',
            r'\bqué más\b',
            r'\bqué hay\b',
            r'\bqué fue\b',
            r'\bque xopa\b',
            r'\bque pasa\b',
            r'\bq tal\b',
            r'\bwenas\b',
            r'\bbuen día\b',
            r'\bgusto en saludarte\b',
            r'\bqué gusto\b',
            r'\bmuy buenas\b',
            r'\bmuy buen[oa]s?\b'
        ]
        
        return any(re.search(p, texto) for p in patrones)
    
    async def fire_and_forget_post(self, action: str, data: dict = None):
        """Enviar un POST asincrónico sin esperar respuesta al endpoint remoto"""
        url = f"https://credible-clam-tolerant.ngrok-free.app/action/{action}"
        try:
            async def post():
                async with httpx.AsyncClient() as client:
                    await client.post(url, json=data or {})
            asyncio.create_task(post())
            logger.info(f"POST async lanzado a {url} con data={data}")
        except Exception as e:
            logger.error(f"Error lanzando POST a {url}: {str(e)}")
