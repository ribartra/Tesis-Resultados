from sqlalchemy import Column, Integer, ForeignKey, DateTime, CheckConstraint
from sqlalchemy.orm import relationship
from app.database import Base
from sqlalchemy.sql import func

class InteractionRating(Base):
    """
    Modelo para ratings de interacción según schema.sql PostgreSQL.
    Permite calificar interacciones entre mensajes de usuario y asistente.
    """
    __tablename__ = 'interaction_ratings'
    
    # Campos según schema.sql
    id_rating = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_msg_id = Column(Integer, ForeignKey('chat_messages.id', ondelete='CASCADE'), nullable=False, index=True)
    assistant_msg_id = Column(Integer, ForeignKey('chat_messages.id', ondelete='CASCADE'), nullable=False, index=True)
    score = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    
    # Constraint para validar el score (1-10)
    __table_args__ = (
        CheckConstraint('score BETWEEN 1 AND 10', name='check_score_range'),
    )
    
    # Relaciones con ChatMessage
    user_message = relationship("ChatMessage", foreign_keys=[user_msg_id], back_populates="user_ratings")
    assistant_message = relationship("ChatMessage", foreign_keys=[assistant_msg_id], back_populates="assistant_ratings")
