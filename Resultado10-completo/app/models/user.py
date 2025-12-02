from sqlalchemy import Column, BigInteger, String, DateTime
from app.database import Base
from sqlalchemy.sql import func

class User(Base):
    """
    Modelo de Usuario.
    
    Nota: Este modelo ya no tiene relaciones con ChatThread o ChatMessage,
    ya que se simplificó el sistema para no requerir autenticación de usuarios.
    Si en el futuro se necesita vincular usuarios con conversaciones,
    agregar el campo id_user a ChatThread con ForeignKey a users.id
    """
    __tablename__ = 'users'
    
    id = Column(BigInteger, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    email_verified_at = Column(DateTime, default=None)
    password = Column(String(255), nullable=False)
    remember_token = Column(String(100), default=None)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
