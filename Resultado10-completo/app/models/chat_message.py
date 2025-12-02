from sqlalchemy import Column, Integer, ForeignKey, String, Text, DateTime, CheckConstraint
from sqlalchemy.orm import relationship
from app.database import Base
from sqlalchemy.sql import func

class ChatMessage(Base):
    """
    Modelo para mensajes de chat según schema.sql PostgreSQL.
    Alineado con el esquema de base de datos.
    """
    __tablename__ = 'chat_messages'
    
    # Campos según schema.sql
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    thread_id = Column(Integer, ForeignKey('chat_threads.id_thread', ondelete='CASCADE'), nullable=False)
    role = Column(String(20), nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Constraint para validar el rol
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name='check_role'),
    )
    
    # Relación con el modelo ChatThread
    thread = relationship("ChatThread", back_populates="messages")
    
    # Relaciones con InteractionRating
    user_ratings = relationship("InteractionRating", foreign_keys="[InteractionRating.user_msg_id]", back_populates="user_message", cascade="all, delete-orphan")
    assistant_ratings = relationship("InteractionRating", foreign_keys="[InteractionRating.assistant_msg_id]", back_populates="assistant_message", cascade="all, delete-orphan")