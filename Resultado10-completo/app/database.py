import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener las variables de entorno para PostgreSQL
DB_USER = os.getenv("DB_USER", "healthuser")
DB_PASSWORD = os.getenv("DB_PASSWORD", "healthpass")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "healthchat")
DB_PORT = os.getenv("DB_PORT", "5432")

# Configuración de la base de datos PostgreSQL usando psycopg2
SQLALCHEMY_DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Crear el motor de base de datos
# PostgreSQL no requiere connect_args especiales como MySQL
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Verificar conexiones antes de usarlas
    pool_size=10,  # Tamaño del pool de conexiones
    max_overflow=20  # Máximo de conexiones adicionales
)

# Crear una sesión de base de datos
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Crear el base
Base = declarative_base()

# Función para obtener la sesión de base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
