from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from app.api.chat import router as chat_router  # Importar las rutas API para el chat
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import app.whisper as whisper_local

# Crear la aplicación FastAPI
app = FastAPI()

# Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite solicitudes de todos los orígenes. Cambia esto si quieres restringirlo.
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
    expose_headers=["Audio-Type","Content-Disposition"
    ]
)

# Configuración para servir archivos estáticos (CSS, JS, imágenes)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Configuración para las plantillas de Jinja2
templates = Jinja2Templates(directory="app/templates")

# Ruta para la página de chat interna
@app.get("/chat")
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

# Ruta para la página principal
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Incluir las rutas de la API para el chat
app.include_router(chat_router, prefix="/api", tags=["chat"])

# Evento de startup explícito para asegurar que whisper cargue al iniciar
@app.on_event("startup")
async def startup_event():
    # Aquí aseguras explícitamente que el modelo esté cargado
    print("Cargando Whisper al iniciar el servidor...")
    _ = whisper_local.whisper_pipeline
    print("Whisper cargado y listo para usarse.")
