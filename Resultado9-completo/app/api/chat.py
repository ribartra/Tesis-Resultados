from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.services.chat_service import ChatService  # Importamos la lógica de negocio
from app.database import get_db  # Abrir sesión de base de datos
from pydantic import BaseModel
from typing import List
from app.models.chat_thread import ChatThread
import base64
from fastapi.responses import JSONResponse, StreamingResponse
import json
import asyncio
import logging
import tempfile
import os
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat_api")

router = APIRouter(prefix="/chat", tags=["chat"])  # Definir el prefijo para las rutas de chat

# Definir los modelos de datos para las solicitudes
class ChatRequest(BaseModel):
    message: str

class ChatRequest2(BaseModel):
    thread_id: int
    message: str

class ChatThreadCreateResponse(BaseModel):
    id_thread: int
    title: str

class ChatResponse(BaseModel):
    message: str
    timing: dict

class ChatResponse2(BaseModel):
    id_thread: int  # Agregar el ID del hilo
    title: str      # Agregar el título del hilo
    message: str    # Respuesta del asistente
    timing: dict    # Información sobre el tiempo de la solicitud

class CreateChatRequest(BaseModel):
    user_id: int = None

class EmptyRequest(BaseModel):
    pass

class SpeechToTextResponse(BaseModel):
    text: str

# Crear una instancia de ChatService para interactuar con la lógica del chatbot
chat_service = ChatService()

# Endpoint para procesar audio y convertirlo a texto usando Whisper
@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(audio_file: UploadFile = File(...)):
    try:
        # Generar archivo temporal
        temp_audio_path = await chat_service.generate_tempfile(audio_file)
        # Transcribir el audio usando el servicio de Whisper
        transcription = chat_service.transcribe_openai(temp_audio_path)            
        return {"text": transcription}
    except Exception as e:
        logger.error(f"Error in Whisper transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la transcripción: {str(e)}")
               
@router.post("/local-speech-to-text")
async def local_speech_to_text(audio_file: UploadFile = File(...)):
    try:
        # Generar archivo temporal
        temp_audio_path = await chat_service.generate_tempfile(audio_file)
        # Utilizar el método de transcripción local
        transcription = chat_service.transcribe_local(temp_audio_path)
        logger.info(f"Transcription successful: '{transcription[:50]}...' if len > 50")
        return {"text": transcription}
    except Exception as e:
        logger.error(f"Error in speech-to-text: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en transcripción local: {str(e)}")

@router.post("/send-audio")
async def send_message_with_audio(request: ChatRequest2, db: Session = Depends(get_db)):
    try:
        logger.info(f"Received request for audio generation for thread {request.thread_id}")
        response = chat_service.send_message_to_thread(db=db, thread_id=request.thread_id, user_message=request.message)
        voz = chat_service.generate_voice(response['message'])
        logger.info(f"Generated audio response for thread {request.thread_id}")

        # Instead of returning JSON with base64, return the audio bytes directly
        return StreamingResponse(
            BytesIO(voz),
            media_type="audio/wav",
            headers={
                'Accept-Ranges': 'bytes',
                'Content-Disposition': 'inline',
                'Text-Content': response['message'],  # Include the text in a header
                'Timing-Info': json.dumps(response['timing'])  # Include timing info in a header
            }
        )
    except Exception as e:
        logger.error(f"Error in /send-audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para obtener la lista de hilos
@router.get("/threads", response_model=List[ChatThreadCreateResponse])
async def get_threads(EmptyRequest = None, db: Session = Depends(get_db)):
    try:
        logger.info("Getting all chat threads")
        threads = db.query(ChatThread).order_by(ChatThread.created_at.desc()).all()
        logger.info(f"Found {len(threads) if threads else 0} threads")
        if threads:
            return [{"id_thread": thread.id_thread, "title": thread.title} for thread in threads]
        else:
            return []  # Devuelve una lista vacía si no hay hilos
    except Exception as e:
        logger.error(f"Error in /threads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para crear un nuevo hilo
@router.post("/threads/create", response_model=ChatThreadCreateResponse)
async def create_thread(EmptyRequest = None, db: Session = Depends(get_db)):
    try:
        logger.info("Creating new chat thread")
        chat_thread = chat_service.create_thread(db=db, user_id=None)
        logger.info(f"Thread created with ID: {chat_thread.id_thread}")
        return {"id_thread": chat_thread.id_thread, "title": chat_thread.title}
    except Exception as e:
        logger.error(f"Error in /threads/create: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para crear un nuevo hilo y ejecutar el primer mensaje con OpenAI
@router.post("/threads/create-run", response_model=ChatResponse2)
async def create_and_run_thread(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"Creating and running thread with message: '{request.message[:30]}...' if len > 30")
        response = chat_service.create_and_run_thread(db=db, user_message=request.message, user_id=None)
        logger.info(f"Thread created and run with ID: {response['id_thread']}")
        return response
    except Exception as e:
        logger.error(f"Error in /threads/create-run: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para guardar el historial de un hilo
@router.post("/threads/save", response_model=dict)
async def save_thread_history(request: dict, db: Session = Depends(get_db)):
    try:
        thread_id = request.get('threadId')
        messages = request.get('messages')
        logger.info(f"Saving history for thread {thread_id} with {len(messages)} messages")
        result = chat_service.save_thread_history(db=db, thread_id=thread_id, messages=messages)
        logger.info(f"History saved for thread {thread_id}")
        return result
    except Exception as e:
        logger.error(f"Error in /threads/save: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para obtener el historial de un hilo específico
@router.get("/history/{thread_id}", response_model=List[dict])
async def get_thread_history(thread_id: int, db: Session = Depends(get_db)):
    try:
        logger.info(f"Getting history for thread {thread_id}")
        history = chat_service.get_thread_history(db=db, thread_id=thread_id)
        logger.info(f"Retrieved {len(history)} messages from thread {thread_id}")
        return history
    except Exception as e:
        logger.error(f"Error in /history/{thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para enviar un mensaje dentro de un hilo
@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest2, db: Session = Depends(get_db)):
    try:
        thread_id = request.thread_id  # El `thread_id` debe ser enviado con el request
        if not thread_id:
            logger.error("Missing thread_id in request")
            raise HTTPException(status_code=400, detail="El ID del hilo es requerido.")

        logger.info(f"Sending message to thread {thread_id}: '{request.message[:30]}...' if len > 30")
        response = chat_service.send_message_to_thread(db=db, thread_id=thread_id, user_message=request.message)
        logger.info(f"Message sent to thread {thread_id}, response time: {response['timing']['total_request_time']}ms")
        return response  # Retorna la respuesta del asistente junto con el `thread_id`
    except Exception as e:
        logger.error(f"Error in /send: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# # Ruta para enviar mensajes de chat con streaming
@router.post("/send-message-stream")
async def send_message_stream(request: ChatRequest2, db: Session = Depends(get_db)):
    try:
        thread_id = request.thread_id
        if not thread_id:
            logger.error("Missing thread_id in stream request")
            raise HTTPException(status_code=400, detail="El ID del hilo es requerido.")
        
        logger.info(f"Starting streaming response for thread {thread_id}: '{request.message[:30]}...' if len > 30")
        
        # Crear un generador para el streaming
        async def event_generator():
            try:
                # event_count = 0
                yield f"data: {json.dumps({'type': 'text_delta', 'text': ''})}\n\n"
                async for event in chat_service.send_message_to_thread_stream(db=db, thread_id=thread_id, user_message=request.message):
                    # event_count += 1
                    # if event_count % 10 == 0:
                    #     logger.debug(f"Sent {event_count} streaming events so far")                    
                    # El evento ya viene con formato "data: {...}\n\n"
                    # Para evitar la doble codificación, verificamos si ya tiene el formato correcto
                    if event.startswith('data: '):
                        yield event
                    else:
                        yield f"data: {event}\n\n"
                    
                    # Forzar el flush añadiendo un espacio en blanco y salto de línea extra
                    yield " \n"
                
                # logger.info(f"Completed streaming response for thread {thread_id}, sent {event_count} events")
            except Exception as e:
                logger.error(f"Error in event_generator: {str(e)}")
                # Enviar un evento de error
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                # Forzar flush con un espacio en blanco adicional
                yield " \n"
        # Devolver una respuesta de streaming
        logger.info(f"Starting StreamingResponse for thread {thread_id}")
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache, no-transform',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # Desactivar buffer de nginx
                'Transfer-Encoding': 'chunked'  # Asegurar transferencia por chunks
            }
        )
    except Exception as e:
        logger.error(f"Error in /send-message-stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Ruta para enviar mensajes de chat con streaming
# @router.post("/send-message-stream")
# async def send_message_stream(request: ChatRequest2, db: Session = Depends(get_db)):
#     try:
#         thread_id = request.thread_id
#         if not thread_id:
#             logger.error("Missing thread_id in stream request")
#             raise HTTPException(status_code=400, detail="El ID del hilo es requerido.")
        
#         logger.info(f"Starting streaming response for thread {thread_id}: '{request.message[:30]}...' if len > 30")

#         async def event_generator():
#             try:
#                 formatted_messages = [
#                     {'role': 'user', 'content': request.message} 
#                 ]
#                 stream = client.responses.create(
#                     model="gpt-4-turbo-preview",
#                     input=formatted_messages,
#                     temperature=0.1,
#                     top_p=0.2,
#                     store=False,
#                     # tools=[{
#                     #     "type": "file_search",
#                     #     "vector_store_ids": ["vs_67fd2551daac819188047bdb1d98b8c8"]
#                     # }],
#                     stream=True
#                 )
#                 complete_response = ""
#                 for event in stream:
#                     if event.type != "response.output_text.delta":
#                         logger.info(f"Processing event : {event.type}")
#                     if event.type == "response.output_text.delta":
#                         delta_text = event.delta

#                         # Filtrar texto que solo contiene caracteres decorativos
#                         if not all(c in ['#', '*', '-'] for c in delta_text.strip()):
#                             complete_response += delta_text
#                         logger.info(delta_text)
#                         yield f"data: {json.dumps({'type': 'text_delta', 'text': delta_text})}\n\n"
#                     yield f": keep-alive\n\n"
#                     await asyncio.sleep(0.01)
#             except Exception as e:
#                 logger.error(f"Error in event_generator: {str(e)}")
#                 # Enviar un evento de error
#                 yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
#                 # Forzar flush con un espacio en blanco adicional
#                 yield " \n"        
#         # Devolver una respuesta de streaming
#         logger.info(f"Starting StreamingResponse for thread {thread_id}")
#         return StreamingResponse(
#             event_generator(),
#             media_type="text/event-stream",
#             headers={
#                 'Content-Type': 'text/event-stream',
#                 'Cache-Control': 'no-cache, no-transform',
#                 'Connection': 'keep-alive',
#                 'X-Accel-Buffering': 'no',  # Desactivar buffer de nginx
#                 'Transfer-Encoding': 'chunked'  # Asegurar transferencia por chunks
#             }
#         )
#     except Exception as e:
#         logger.error(f"Error in /send-message-stream: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/send-message-text-stream")
async def send_message_text_stream(request: ChatRequest2, db: Session = Depends(get_db)):
    try:
        thread_id = request.thread_id
        if not thread_id:
            logger.error("Missing thread_id in text stream request")
            raise HTTPException(status_code=400, detail="El ID del hilo es requerido.")
        
        logger.info(f"Starting text-only streaming response for thread {thread_id}: '{request.message[:30]}...' if len > 30")
        
        async def event_generator():
            try:
                event_count = 0
                async for event in chat_service.send_message_to_thread_text_stream(
                    db=db, 
                    thread_id=thread_id, 
                    user_message=request.message
                ):
                    event_count += 1
                    if event_count % 10 == 0:
                        logger.debug(f"Sent {event_count} text streaming events so far")
                        
                    # El evento ya viene con formato "data: {...}\n\n"
                    if event.startswith('data: '):
                        yield event
                    else:
                        yield f"data: {event}\n\n"
                    
                    # Forzar el flush añadiendo un espacio en blanco y salto de línea extra
                    yield " \n"
                
                logger.info(f"Completed text-only streaming for thread {thread_id}, sent {event_count} events")
            except Exception as e:
                logger.error(f"Error in text stream event_generator: {str(e)}")
                # Enviar un evento de error
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                # Forzar flush con un espacio en blanco adicional
                yield " \n"
        
        logger.info(f"Starting text-only StreamingResponse for thread {thread_id}")
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache, no-transform',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # Desactivar buffer de nginx
                'Transfer-Encoding': 'chunked'  # Asegurar transferencia por chunks
            }
        )
    except Exception as e:
        logger.error(f"Error in /send-message-text-stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Ruta para crear un hilo nuevo y enviar el primer mensaje con streaming
@router.post("/threads/create-run-stream")
async def create_and_run_thread_stream(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        logger.info(f"Creating thread with streaming response: '{request.message[:30]}...' if len > 30")
        # yield f"data: {json.dumps({'type': 'text_delta', 'text': 'Start:'})}\n\n"
        # # Enviar un comentario vacío para forzar el flush
        # yield f": keep-alive\n\n"
        # Crear un generador para el streaming
        async def event_generator():
            try:
                event_count = 0
                thread_id = None
                logger.info(f"Start stream generator")        
                yield f"data: {json.dumps({'type': 'text_delta', 'text': 'Start:'})}\n\n"
                yield f": keep-alive\n\n"


                async for event in chat_service.create_and_run_thread_stream(db=db, user_message=request.message, user_id=None):
                    event_count += 1
                    
                    # Extraer thread_id para logging si es un evento de creación de hilo
                    if event.startswith('data: ') and 'thread_created' in event:
                        try:
                            data = json.loads(event[6:].strip())
                            if data.get('type') == 'thread_created':
                                thread_id = data.get('id_thread')
                                logger.info(f"New thread created with ID: {thread_id}")
                        except:
                            pass
                    
                    if event_count % 10 == 0:
                        logger.debug(f"Sent {event_count} streaming events for new thread {thread_id or 'unknown'}")
                    
                    # El evento ya viene con formato "data: {...}\n\n"
                    if event.startswith('data: '):
                        yield event
                    else:
                        yield f"data: {event}\n\n"
                    
                    # Forzar el flush añadiendo un espacio en blanco y salto de línea extra
                    yield " \n"
                
                logger.info(f"Completed streaming for new thread {thread_id or 'unknown'}, sent {event_count} events")
            except Exception as e:
                logger.error(f"Error in create_and_run event_generator: {str(e)}")
                # Enviar un evento de error
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                # Forzar flush con un espacio en blanco adicional
                yield " \n"
        
        # Devolver una respuesta de streaming
        logger.info("Starting StreamingResponse for new thread creation")
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache, no-transform',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',  # Desactivar buffer de nginx
                'Transfer-Encoding': 'chunked'  # Asegurar transferencia por chunks
            }
        )
    except Exception as e:
        logger.error(f"Error in /threads/create-run-stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))