from sqlalchemy import Column, Integer, DateTime
from sqlalchemy.orm import relationship
from app.database import Base
from sqlalchemy.sql import func

class ChatThread(Base):
    """
    Modelo para hilos de chat según schema.sql PostgreSQL.
    Simplificado para alinearse con el esquema de base de datos.
    """
    __tablename__ = 'chat_threads'
    
    # Campos según schema.sql
    id_thread = Column(Integer, primary_key=True, index=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    
    # Relación con el modelo ChatMessage
    messages = relationship("ChatMessage", back_populates="thread", cascade="all, delete-orphan")