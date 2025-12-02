from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.chat_service import ChatService
from app.database import get_db
from pydantic import BaseModel
from typing import List
from app.models.chat_thread import ChatThread
from fastapi.responses import StreamingResponse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat_api")

router = APIRouter(prefix="/chat", tags=["chat"])

# ========== MODELOS PYDANTIC ==========

class ChatRequest(BaseModel):
    """Request para enviar un mensaje"""
    thread_id: int
    message: str

class ChatThreadResponse(BaseModel):
    """Response con info de un thread"""
    id_thread: int

class InteractionRatingRequest(BaseModel):
    """Request para calificar una interacción"""
    user_msg_id: int
    assistant_msg_id: int
    score: int  # 1-10

class InteractionRatingResponse(BaseModel):
    """Response de un rating creado"""
    id_rating: int
    user_msg_id: int
    assistant_msg_id: int
    score: int

# ========== INSTANCIA DE SERVICIO ==========

chat_service = ChatService()

# ========== ENDPOINTS DE THREADS ==========

@router.get("/threads", response_model=List[ChatThreadResponse])
async def get_threads(db: Session = Depends(get_db)):
    """Obtener lista de todos los hilos"""
    try:
        logger.info("Getting all chat threads")
        threads = chat_service.get_threads(db)
        logger.info(f"Found {len(threads)} threads")
        return [{"id_thread": thread.id_thread} for thread in threads]
    except Exception as e:
        logger.error(f"Error in /threads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/threads/create", response_model=ChatThreadResponse)
async def create_thread(db: Session = Depends(get_db)):
    """Crear un nuevo hilo de conversación"""
    try:
        logger.info("Creating new chat thread")
        thread = chat_service.create_thread(db)
        logger.info(f"Thread created with ID: {thread.id_thread}")
        return {"id_thread": thread.id_thread}
    except Exception as e:
        logger.error(f"Error in /threads/create: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{thread_id}", response_model=List[dict])
async def get_thread_history(thread_id: int, db: Session = Depends(get_db)):
    """Obtener historial de mensajes de un hilo"""
    try:
        logger.info(f"Getting history for thread {thread_id}")
        history = chat_service.get_thread_history(db, thread_id)
        logger.info(f"Retrieved {len(history)} messages")
        return history
    except Exception as e:
        logger.error(f"Error in /history/{thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== ENDPOINT PRINCIPAL: CHAT CON RAG ==========

@router.post("/send-message-rag-advanced-stream")
async def send_message_rag_advanced_stream(
    request: ChatRequest, 
    db: Session = Depends(get_db)
):
    """
    Enviar mensaje con RAG Advanced en streaming.
    
    Flujo:
    1. Recupera fuentes (top 5 documentos)
    2. Genera respuesta en streaming
    3. Devuelve IDs para rating posterior
    """
    try:
        thread_id = request.thread_id
        if not thread_id:
            logger.error("Missing thread_id in request")
            raise HTTPException(status_code=400, detail="El ID del hilo es requerido.")
        
        logger.info(f"Starting RAG Advanced stream for thread {thread_id}")
        
        async def event_generator():
            try:
                async for event in chat_service.send_message_rag_advanced_stream(
                    db=db, 
                    thread_id=thread_id, 
                    user_message=request.message
                ):
                    if event.startswith('data: '):
                        yield event
                    else:
                        yield f"data: {event}\n\n"
                    yield " \n"  # Force flush
                
            except Exception as e:
                logger.error(f"Error in event_generator: {str(e)}")
                yield f"data: {{'type': 'error', 'message': '{str(e)}'}}\n\n"
                yield " \n"
        
        logger.info(f"Starting StreamingResponse for thread {thread_id}")
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache, no-transform',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Transfer-Encoding': 'chunked'
            }
        )
    except Exception as e:
        logger.error(f"Error in /send-message-rag-advanced-stream: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== ENDPOINTS DE RATINGS ==========

@router.post("/ratings/create", response_model=InteractionRatingResponse)
async def create_interaction_rating(
    request: InteractionRatingRequest,
    db: Session = Depends(get_db)
):
    """
    Crear un rating para una interacción usuario-asistente.
    
    Args:
        user_msg_id: ID del mensaje del usuario
        assistant_msg_id: ID del mensaje del asistente
        score: Puntuación de 1 a 10
    """
    try:
        logger.info(f"Creating rating: score={request.score}")
        
        # Validar score
        if request.score < 1 or request.score > 10:
            raise HTTPException(
                status_code=400, 
                detail="El score debe estar entre 1 y 10"
            )
        
        rating = chat_service.create_interaction_rating(
            db=db,
            user_msg_id=request.user_msg_id,
            assistant_msg_id=request.assistant_msg_id,
            score=request.score
        )
        
        logger.info(f"Rating created with ID: {rating.id_rating}")
        return {
            "id_rating": rating.id_rating,
            "user_msg_id": rating.user_msg_id,
            "assistant_msg_id": rating.assistant_msg_id,
            "score": rating.score
        }
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /ratings/create: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ratings/{thread_id}", response_model=List[dict])
async def get_thread_ratings(thread_id: int, db: Session = Depends(get_db)):
    """Obtener todos los ratings de un hilo"""
    try:
        logger.info(f"Getting ratings for thread {thread_id}")
        ratings = chat_service.get_interaction_ratings(db, thread_id)
        logger.info(f"Found {len(ratings)} ratings")
        return ratings
    except Exception as e:
        logger.error(f"Error in /ratings/{thread_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ratings", response_model=List[dict])
async def get_all_ratings(db: Session = Depends(get_db)):
    """Obtener todos los ratings del sistema"""
    try:
        logger.info("Getting all ratings")
        ratings = chat_service.get_interaction_ratings(db)
        logger.info(f"Found {len(ratings)} ratings")
        return ratings
    except Exception as e:
        logger.error(f"Error in /ratings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
