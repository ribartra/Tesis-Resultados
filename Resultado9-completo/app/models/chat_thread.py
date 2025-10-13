from sqlalchemy import Column, Integer, BigInteger, String, ForeignKey, DateTime  # Usamos BigInteger
from sqlalchemy.orm import relationship
from app.database import Base
from sqlalchemy.sql import func

class ChatThread(Base):
    __tablename__ = 'chat_threads'
    
    id_thread = Column(Integer, primary_key=True, index=True)
    code = Column(String(255), nullable=True)
    id_user = Column(BigInteger, ForeignKey('users.id'), nullable=True)
    category = Column(String(50), default='qhali-llama')
    title = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())  # Usar server_default para func.now()
    updated_at = Column(DateTime, onupdate=func.now(), server_default=func.now())    
    
    # Relación con el modelo ChatMessage
    messages = relationship("ChatMessage", back_populates="thread")

    # Relación con el modelo User
    user = relationship("User", back_populates="chat_threads")