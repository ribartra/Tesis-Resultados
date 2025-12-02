"""
FastAPI Application - Qhali Health Chat
Simplified version: RAG + Conversations + Ratings
No OpenAI, No Audio
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import logging

# Importar routers
from app.api.chat import router as chat_router
from app.database import Base, engine

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

# Crear tablas si no existen (solo en desarrollo)
# En producción, usar Alembic para migraciones
try:
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables verified/created")
except Exception as e:
    logger.warning(f"Could not create tables: {e}")

# Crear aplicación FastAPI
app = FastAPI(
    title="Qhali Health Chat API",
    description="Chat con RAG Advanced para promoción de salud",
    version="2.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Incluir routers con prefijo /api
app.include_router(chat_router, prefix="/api")

# Ruta raíz - Servir la interfaz de chat
@app.get("/", response_class=HTMLResponse)
async def root():
    """Servir página principal de chat"""
    template_path = Path("app/templates/chat.html")
    if template_path.exists():
        return template_path.read_text()
    else:
        return """
        <html>
            <head><title>Qhali Chat</title></head>
            <body>
                <h1>Qhali Health Chat</h1>
                <p>Interfaz no encontrada. Asegúrate de que app/templates/chat.html existe.</p>
                <p>API disponible en: <a href="/docs">/docs</a></p>
            </body>
        </html>
        """

# Health check endpoint
@app.get("/health")
async def health_check():
    """Endpoint para verificar que el servicio está activo"""
    return {
        "status": "healthy",
        "service": "qhali-health-chat",
        "version": "2.0.0"
    }

# Información de la API
@app.get("/info")
async def api_info():
    """Información sobre la API"""
    return {
        "name": "Qhali Health Chat API",
        "version": "2.0.0",
        "description": "API simplificada para chat con RAG Advanced",
        "features": [
            "Conversaciones con hilos (threads)",
            "RAG Advanced con recuperación de fuentes",
            "Rating de interacciones (1-10)",
            "Streaming de respuestas",
            "PostgreSQL con Docker"
        ],
        "endpoints": {
            "chat": "/api/chat/*",
            "docs": "/docs",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
