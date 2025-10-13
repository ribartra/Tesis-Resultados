from sqlalchemy import Column, Integer, BigInteger, ForeignKey, String, Text, DateTime  # Usamos BigInteger
from sqlalchemy.orm import relationship
from app.database import Base
from sqlalchemy.sql import func

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey('chat_threads.id_thread'), nullable=False)
    user_id = Column(BigInteger, ForeignKey('users.id'), nullable=True)  # Cambiado a BigInteger
    message = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    created_at = Column(DateTime, server_default=func.now())  # Usar server_default para func.now()
    updated_at = Column(DateTime, onupdate=func.now(), server_default=func.now())    
    
    # Relación con el modelo ChatThread
    thread = relationship("ChatThread", back_populates="messages")

    # Relación con el modelo User
    user = relationship("User", back_populates="chat_messages")