from sqlalchemy import Column, BigInteger, String, DateTime
from sqlalchemy.orm import relationship
from app.database import Base
from sqlalchemy.sql import func

class User(Base):
    __tablename__ = 'users'
    
    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    email_verified_at = Column(DateTime, default=None)
    password = Column(String(255), nullable=False)
    remember_token = Column(String(100), default=None)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relaciones con otras tablas
    chat_threads = relationship("ChatThread", back_populates="user")
    chat_messages = relationship("ChatMessage", back_populates="user")
