from sqlalchemy.orm import Session
from app.models.chat_thread import ChatThread
from app.models.chat_message import ChatMessage
from app.models.interaction_rating import InteractionRating
from app.rag_agent import RAGAgent
from typing import List
import logging
import time
import asyncio
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat_service")


class ChatService:
    """
    Servicio simplificado de chat con RAG Advanced.
    Solo incluye lo esencial: conversación, recuperación con fuentes y ratings.
    """
    
    def __init__(self):
        self.timings = {}
        self._inicializar_rag_agent()
    
    def _inicializar_rag_agent(self):
        """Inicializar el agente RAG local con modo advanced"""
        try:
            self.rag_agent = RAGAgent(
                model_id="Llama-3.2-3B-Instruct_Q4_K_M:latest",
                embedder_id="nomic-embed-text-v2",
                embedding_dim=768,
                lancedb_path="tmp/lancedb",
                table_name="docs_qa",
                translator_model_dir="nllb200_600M_int8",
                max_history=10,
                top_k_semantic=5,
                top_k_keyword=5,
                alpha=0.55,  # Híbrido: 55% semántico + 45% keyword
                similarity_threshold=0.7,
                show_tool_calls=False,
                markdown=False,
                auto_warm_up=True,  # Warm-up habilitado
                enable_translation=True,
                silent_translation=True,
                skip_chunk_translation=False,
                # Parámetros Advanced RAG
                mode="advanced",
                adv_num_queries=3,
                adv_top_k_per_query=5,
                adv_merge_strategy="vote+score",
                adv_rerank_strategy="mmr",
                adv_max_chunks=6
            )
            logger.info("RAG Agent inicializado correctamente en modo advanced")
            logger.info(f"Configuración: model={self.rag_agent.model_id}, mode={self.rag_agent.mode}")
        except Exception as e:
            logger.error(f"Error inicializando RAG Agent: {e}")
            self.rag_agent = None
    
    def create_thread(self, db: Session) -> ChatThread:
        """Crear un nuevo hilo de conversación"""
        logger.info("Creating new thread")
        
        # Limpiar el historial del agente RAG para nueva conversación
        if self.rag_agent:
            self.rag_agent.limpiar_historial()
            logger.info("RAG Agent history cleared for new thread")
        
        new_thread = ChatThread()
        db.add(new_thread)
        db.commit()
        db.refresh(new_thread)
        logger.info(f"Thread created with ID: {new_thread.id_thread}")
        return new_thread
    
    def get_threads(self, db: Session) -> List[ChatThread]:
        """Obtener todos los hilos ordenados por fecha"""
        threads = db.query(ChatThread).order_by(ChatThread.created_at.desc()).all()
        return threads
    
    def get_thread_history(self, db: Session, thread_id: int) -> List[dict]:
        """Obtener historial de mensajes de un hilo"""
        try:
            messages = db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread_id
            ).order_by(ChatMessage.created_at.asc()).all()
            
            return [{"role": msg.role, "message": msg.message} for msg in messages]
        except Exception as e:
            logger.error(f"Error getting thread history {thread_id}: {str(e)}")
            raise Exception(f"Error al cargar el historial del hilo {thread_id}.")
    
    async def send_message_rag_advanced_stream(
        self, 
        db: Session, 
        thread_id: int, 
        user_message: str
    ):
        """
        Enviar mensaje con RAG Advanced en streaming.
        Devuelve: fuentes (top 5) + respuesta generada en streaming
        """
        self.timings['start_request'] = time.time() * 1000
        logger.info(f"Starting RAG Advanced stream for thread {thread_id}")

        # Inicializar con evento vacío para abrir el stream
        yield f"data: {json.dumps({'type': 'text_delta', 'text': ''})}\n\n"
        yield f": keep-alive\n\n"

        # Guardar mensaje del usuario
        user_message_record = ChatMessage(
            thread_id=thread_id,
            message=user_message,
            role="user"
        )
        db.add(user_message_record)
        db.commit()
        logger.info(f"User message saved to database for thread {thread_id}")
        
        try:
            # Recuperar historial para contexto (últimos 15 mensajes)
            messages = db.query(ChatMessage).filter(
                ChatMessage.thread_id == thread_id
            ).order_by(ChatMessage.created_at.desc()).limit(15).all()
            
            # Convertir a formato de historial para RAG
            historial_conversacional = self.rag_agent.establecer_historial_desde_bd(
                list(reversed(messages))
            )
            
            # PASO 1: Recuperar fuentes y agrupar por documento
            logger.info("Retrieving sources with RAG Advanced")
            retrieved_chunks = self.rag_agent.retrieve_only(user_message)
            
            sources_list = []  # Inicializar la lista aquí
            grouped_sources = {}
            if retrieved_chunks:
                # Agrupar chunks por documento
                for chunk in retrieved_chunks[:10]:  # Tomar más chunks para agrupar
                    pdf_name = chunk.get("pdf_name", "Desconocido")
                    
                    if pdf_name not in grouped_sources:
                        grouped_sources[pdf_name] = {
                            "pdf_name": pdf_name,
                            "titulo": chunk.get("titulo", ""),
                            "fuente": chunk.get("fuente", ""),
                            "chunks": []
                        }
                    
                    # Agregar chunk al documento con su texto
                    chunk_text = chunk.get("chunk_text", chunk.get("text", ""))
                    # Truncar si es muy largo (máximo 200 caracteres)
                    if len(chunk_text) > 200:
                        chunk_text = chunk_text[:200] + "..."
                    
                    grouped_sources[pdf_name]["chunks"].append({
                        "chunk_index": chunk.get("chunk_index", 0),
                        "score": round(chunk.get("score_final", 0.0), 4),
                        "chunk_id": chunk.get("chunk_id", ""),
                        "text": chunk_text
                    })
                
                # Convertir a lista y ordenar por score máximo
                sources_list = []
                for doc in grouped_sources.values():
                    # Calcular score promedio del documento
                    avg_score = sum(c["score"] for c in doc["chunks"]) / len(doc["chunks"]) if doc["chunks"] else 0
                    doc["avg_score"] = round(avg_score, 4)
                    doc["max_score"] = max(c["score"] for c in doc["chunks"]) if doc["chunks"] else 0
                    doc["num_chunks"] = len(doc["chunks"])
                    sources_list.append(doc)
                
                # Ordenar por score promedio descendente y tomar top 5 documentos
                sources_list.sort(key=lambda x: x["avg_score"], reverse=True)
                sources_list = sources_list[:5]
                
                logger.info(f"Retrieved {len(sources_list)} documents with {sum(d['num_chunks'] for d in sources_list)} chunks")
                
                # Enviar fuentes agrupadas
                yield f"data: {json.dumps({'type': 'sources', 'sources': sources_list})}\n\n"
                yield f": keep-alive\n\n"
            
            # PASO 2: Generar respuesta con streaming
            logger.info("Generating response with RAG Advanced")
            
            complete_response = ""
            token_count = 0

            # Generar respuesta - SOLO capturar la generación del LLM, NO los chunks recuperados
            for rag_event in self.rag_agent.responder_con_rag_streaming_avanzado(
                user_message, 
                historial_conversacional
            ):
                # Solo capturar la generación real del LLM (no los chunks de retrieval)
                if rag_event["type"] in ["generation", "direct_answer"]:
                    delta_text = rag_event["content"]
                    
                    if delta_text and delta_text.strip():
                        complete_response += delta_text
                        yield f"data: {json.dumps({'type': 'text_delta', 'text': delta_text})}\n\n"
                        yield f": keep-alive\n\n"
                        await asyncio.sleep(0.01)
                        token_count += 1
                
                elif rag_event["type"] == "completion":
                    logger.info("RAG Advanced streaming completed")
                    break
            
            # Guardar respuesta del asistente
            assistant_message = ChatMessage(
                thread_id=thread_id,
                message=complete_response,
                role="assistant"
            )
            db.add(assistant_message)
            db.commit()
            db.refresh(assistant_message)
            
            # Guardar IDs para rating posterior
            assistant_msg_id = assistant_message.id
            
            logger.info(f"Assistant message saved with ID: {assistant_msg_id}")
            
            # Métricas finales
            self.timings['total_request_time'] = (time.time() * 1000) - self.timings['start_request']
            logger.info(f"Stream completed in {self.timings['total_request_time']:.2f}ms")
            
            # Enviar completion con metadata
            completion_data = {
                'type': 'completion',
                'thread_id': thread_id,
                'user_msg_id': user_message_record.id,
                'assistant_msg_id': assistant_msg_id,
                'timing': self.timings,
                'sources_count': len(sources_list)
            }
            yield f"data: {json.dumps(completion_data)}\n\n"
            
        except Exception as e:
            logger.error(f"Error in RAG Advanced stream: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    def create_interaction_rating(
        self,
        db: Session,
        user_msg_id: int,
        assistant_msg_id: int,
        score: int
    ) -> InteractionRating:
        """
        Crear un rating para una interacción.
        Score: 1-10
        """
        logger.info(f"Creating rating: user_msg={user_msg_id}, assistant_msg={assistant_msg_id}, score={score}")
        
        # Validar que los mensajes existen
        user_msg = db.query(ChatMessage).filter(ChatMessage.id == user_msg_id).first()
        assistant_msg = db.query(ChatMessage).filter(ChatMessage.id == assistant_msg_id).first()
        
        if not user_msg or not assistant_msg:
            raise ValueError("Mensaje de usuario o asistente no encontrado")
        
        # Crear rating
        rating = InteractionRating(
            user_msg_id=user_msg_id,
            assistant_msg_id=assistant_msg_id,
            score=score
        )
        db.add(rating)
        db.commit()
        db.refresh(rating)
        
        logger.info(f"Rating created with ID: {rating.id_rating}")
        return rating
    
    def get_interaction_ratings(self, db: Session, thread_id: int = None) -> List[dict]:
        """Obtener ratings de interacciones, opcionalmente filtrados por thread"""
        query = db.query(InteractionRating).join(
            ChatMessage, InteractionRating.user_msg_id == ChatMessage.id
        )
        
        if thread_id:
            query = query.filter(ChatMessage.thread_id == thread_id)
        
        ratings = query.all()
        
        return [{
            "id_rating": r.id_rating,
            "user_msg_id": r.user_msg_id,
            "assistant_msg_id": r.assistant_msg_id,
            "score": r.score,
            "created_at": r.created_at.isoformat()
        } for r in ratings]
