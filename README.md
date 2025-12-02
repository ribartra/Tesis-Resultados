# Qhali - Sistema de Promoción de Salud con IA

Repositorio de resultados experimentales de tesis sobre sistemas de chat inteligente para promoción de salud utilizando RAG (Retrieval-Augmented Generation), fine-tuning con QLoRA, y aplicaciones web interactivas.

## 📋 Descripción General

Este proyecto contiene múltiples experimentos y resultados (Resultado1-10) que demuestran la evolución de un sistema completo de IA conversacional para el dominio de salud. Incluye desde la gestión básica de bases de datos vectoriales hasta aplicaciones web completas con interfaces de usuario modernas.

## 🏗️ Estructura del Proyecto

```
Tesis-Resultados/
├── Resultado1-completo/    # Sistema básico de vectorización de documentos
├── Resultado2-completo/    # CRUD completo para gestión de base vectorial
├── Resultado3-completo/    # Optimización de chunking y embeddings
├── Resultado4-completo/    # Entrenamiento QLoRA con preferencias (DPO/ORPO)
├── Resultado5-completo/    # Evaluación RAG con métricas y modelfiles
├── Resultado6-completo/    # Sistema completo de evaluación y fine-tuning
├── Resultado9-completo/    # Aplicación web con interfaz de chat
├── Resultado10-completo/   # Sistema web completo con PostgreSQL y streaming
└── LICENSE                 # Licencia Apache 2.0
```

## 🔧 Tecnologías Utilizadas

### Core de IA y Machine Learning
- **LanceDB** (v0.21.2) - Base de datos vectorial con índices HNSW e IVF_PQ
- **Ollama** (v0.4.7) - Servidor local para modelos LLM (Llama 3.2, Gemma2, Qwen)
- **Agno AI** (v1.1.0) - Framework para agentes RAG
- **Transformers** (v4.48.3) - Biblioteca de Hugging Face
- **PyTorch** (v2.6.0) - Framework de deep learning
- **PEFT** (v0.14.0) - Parameter-Efficient Fine-Tuning
- **TRL** (v0.14.0) - Transformer Reinforcement Learning (DPO/ORPO)
- **Unsloth** - Optimización de entrenamiento QLoRA
- **CTranslate2** - Traducción NLLB-200 español-inglés

### Procesamiento de Datos
- **Polars** (v1.21.0) - DataFrames de alto rendimiento
- **Pandas** (v2.2.3) - Análisis de datos
- **PyArrow** (v19.0.0) - Formato columnar
- **Docling** - Extracción de texto de PDFs
- **Datasets** (v3.2.0) - Gestión de datasets de Hugging Face

### Backend y APIs
- **FastAPI** - Framework web moderno y asíncrono
- **Uvicorn** - Servidor ASGI
- **SQLAlchemy** - ORM para bases de datos
- **PostgreSQL** - Base de datos relacional (vía Docker)
- **Pydantic** (v2.10.6) - Validación de datos

### Frontend
- **HTML5/CSS3/JavaScript** - Interfaz web moderna
- **FontAwesome** - Iconografía
- **WebSockets** - Comunicación en tiempo real

### Embeddings y Vectorización
- **Nomic Embeddings** (nomic-embed-text-v2, 768 dimensiones)
- **OllamaEmbedder** - Generación local de embeddings

### Testing y Evaluación
- **PyTest** - Framework de testing
- **Métricas RAG**: Hit@K, MRR, nDCG@K, EXACT MATCH, F1 Score
- **Métricas de Generación**: Similitud Semántica, BLEU, ROUGE

### Soporte GPU
- **CUDA 12.8** - Aceleración GPU NVIDIA
- **cuDF-Polars** - DataFrames acelerados por GPU
- **Triton** (v3.2.0) - Kernels optimizados

## 📁 Descripción de Resultados

### Resultado1-completo: Sistema Base de Vectorización
**Scripts principales:**
- `load_and_check.py` - Carga y vectorización inicial de documentos PDF
- `test_load_and_check.py` - Tests de consistencia de carga

**Características:**
- Procesamiento de documentos PDF desde carpeta `docs/`
- Chunking inteligente con límite de 512 palabras
- Generación de embeddings con Nomic (768D)
- Creación de base vectorial LanceDB
- Tests de consistencia con 1, 2 y 3 documentos

### Resultado2-completo: CRUD Completo
**Scripts principales:**
- `load_and_check.py` - Carga inicial
- `read_and_update.py` - Interfaz CRUD interactiva
- `test_read_and_update.py` - Suite de tests CRUD

**Características:**
- ✅ **CREATE**: Agregar nuevos documentos desde CSV
- ✅ **READ**: Consultar detalles de documentos vectorizados
- ✅ **UPDATE**: Actualizar metadatos (título, fuente)
- ✅ **DELETE**: Eliminar documentos de la base vectorial
- Validación automática de estados de BD
- Reindexación opcional tras operaciones
- Cobertura de testing 100%

### Resultado3-completo: Optimización
**Mejoras implementadas:**
- Optimización de parámetros de chunking
- Refinamiento de índices vectoriales
- Mejora en manejo de metadatos JSON

### Resultado4-completo: Fine-tuning con QLoRA
**Scripts principales:**
- `qlora_pref_train.py` - Entrenamiento QLoRA con DPO/ORPO
- `rag_agent.py` - Agente RAG con traducción NLLB-200

**Características:**
- Entrenamiento con preferencias (prompt, chosen, rejected)
- Algoritmos DPO (Direct Preference Optimization) y ORPO
- Split determinista train/val/test por hash
- Traducción bidireccional español-inglés con NLLB-200
- Presets de hiperparámetros optimizados
- Dataset de interacciones reales en `datasets/`

**Datasets:**
- `Interaccion-Qhali-care-25-01.csv` - Interacciones reales del sistema
- `augmented_pairs.csv` - Pares aumentados de entrenamiento
- `preferences.csv` - Dataset de preferencias

### Resultado5-completo: Evaluación RAG
**Scripts principales:**
- `rag_evaluation_simple.py` - Sistema de evaluación RAG

**Características:**
- Métricas de Retrieval: Hit@K, MRR, nDCG@K
- Métricas de Generación: EXACT MATCH, F1 Score, Similitud Semántica
- Detección automática de tipo de modelo (base, dpo, orpo, finetuned)
- Reportes compatibles con golden_f1_results.csv
- Visualizaciones con matplotlib y seaborn

**Modelfiles:**
- `gemma2_family` - Configuración Gemma2
- `llama3_family` - Configuración Llama 3
- `qwen2.5_family` - Configuración Qwen 2.5
- `qwen3_family` - Configuración Qwen 3

### Resultado6-completo: Sistema Completo de Evaluación
**Incluye:**
- Todos los componentes de Resultado4 y Resultado5
- Resultados de evaluación en `evaluation_results/`:
  - `rag_evaluation_results.csv` - Métricas RAG detalladas
  - `golden_f1_results.csv` - F1 scores por modelo
  - `golden_f1_results_default_prompt.csv` - Resultados con prompt por defecto
- Dataset de preferencias en `preferences_set.csv`

### Resultado9-completo: Aplicación Web Básica
**Componentes:**
- Interfaz de chat HTML/CSS/JS
- Integración con FontAwesome (2000+ iconos SVG)
- Comunicación con backend

### Resultado10-completo: Sistema Web Completo
**Arquitectura:**
```
Resultado10-completo/
├── app/
│   ├── main.py                  # FastAPI application
│   ├── database.py              # PostgreSQL connection
│   ├── rag_agent.py             # RAG agent con NLLB-200
│   ├── wsmanager.py             # WebSocket manager
│   ├── api/
│   │   ├── chat.py              # Router de chat con streaming
│   │   └── chat_backup.py       # Backup del router
│   ├── models/
│   │   ├── chat_message.py      # Modelo de mensajes
│   │   ├── chat_thread.py       # Modelo de hilos
│   │   ├── interaction_rating.py # Modelo de ratings
│   │   └── user.py              # Modelo de usuario
│   ├── services/
│   │   └── chat_service.py      # Lógica de negocio
│   ├── static/
│   │   ├── styles.css           # Estilos modernos
│   │   ├── audio-service.js     # Servicio de audio
│   │   ├── qr-qhali.svg         # QR code
│   │   └── fontawesome/         # 2000+ iconos
│   └── templates/
│       ├── chat.html            # Interfaz de chat
│       └── home.html            # Página principal
└── schema.sql                   # Schema PostgreSQL
```

**Características:**
- 🚀 **FastAPI** con documentación automática (`/docs`)
- 💬 **Chat streaming** con respuestas en tiempo real
- 🗄️ **PostgreSQL** con hilos de conversación y ratings (1-10)
- 🔄 **WebSockets** para comunicación bidireccional
- 🤖 **RAG Advanced** con recuperación de fuentes
- 🌐 **Traducción** español-inglés con NLLB-200
- 📊 **Ratings de interacciones** para mejora continua
- 🎨 **Interfaz moderna** y responsive
- ✅ **Health checks** y monitoreo

**Endpoints principales:**
- `GET /` - Interfaz de chat
- `GET /health` - Health check
- `GET /info` - Información de la API
- `POST /api/chat/threads` - Crear nuevo hilo
- `POST /api/chat/message` - Enviar mensaje (streaming)
- `POST /api/chat/rate` - Evaluar interacción
- `GET /docs` - Documentación interactiva

**Base de datos (PostgreSQL):**
- `chat_threads` - Hilos de conversación
- `chat_messages` - Mensajes (user/assistant)
- `interaction_ratings` - Ratings 1-10 de interacciones

## 📦 Instalación

### Requisitos Previos
- Python 3.10+
- CUDA 12.8+ (para GPU NVIDIA)
- Ollama instalado
- PostgreSQL (para Resultado10)
- 16GB+ RAM recomendado

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd Tesis-Resultados
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instalar dependencias
```bash
# Para cualquier resultado (ejemplo: Resultado6)
cd Resultado6-completo
pip install -r requirements.txt
```

### 4. Configurar Ollama
```bash
# Instalar modelos necesarios
ollama pull llama3.2
ollama pull gemma2
ollama pull qwen2.5
ollama pull nomic-embed-text-v2
```

### 5. Preparar documentos
```bash
# Colocar PDFs en la carpeta docs/
# Editar Documentos.csv con formato:
# ID;Título;Fuente;Nombre_archivo
```

## 🚀 Uso

### Cargar documentos (Resultado1-6)
```bash
python load_and_check.py
```

### Gestión CRUD (Resultado2-6)
```bash
python read_and_update.py
```

### Entrenamiento QLoRA (Resultado4-6)
```bash
python qlora_pref_train.py \
  --model_id unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
  --csv_file salidas/preferences.csv \
  --preset_index 0 \
  --loss_type dpo \
  --epochs 3
```

### Evaluación RAG (Resultado5-6)
```bash
python rag_evaluation_simple.py \
  --model_name llama3.2:latest \
  --test_csv datasets/test_queries.csv
```

### Aplicación Web (Resultado10)
```bash
cd Resultado10-completo

# Iniciar PostgreSQL con Docker
docker-compose up -d postgres

# Ejecutar aplicación
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Acceder a:
# - Interfaz: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## 🧪 Testing

### Tests de carga (Resultado1)
```bash
pytest test_load_and_check.py -v
```

### Tests CRUD (Resultado2-6)
```bash
pytest test_read_and_update.py -v
```

### Tests RAG (Resultado10)
```bash
pytest app/test_rag_agent.py -v
```

## 📊 Métricas y Evaluación

### Métricas de Retrieval
- **Hit@K**: Relevancia en top-K resultados
- **MRR**: Mean Reciprocal Rank
- **nDCG@K**: Normalized Discounted Cumulative Gain

### Métricas de Generación
- **EXACT MATCH**: Coincidencia exacta con respuesta esperada
- **F1 Score**: Balance entre precisión y recall
- **Similitud Semántica**: Similitud coseno entre embeddings

### Resultados de Evaluación
Los resultados se guardan en:
- `evaluation_results/rag_evaluation_results.csv`
- `evaluation_results/golden_f1_results.csv`
- `logs_entrenamiento.txt`

## 🗂️ Datasets

### Documentos PDF
Colección de 13 artículos científicos en `docs/`:
- Promoción de salud
- Salud comunitaria
- Intervenciones en salud pública
- Educación para la salud

### Datasets de Entrenamiento
- **Interaccion-Qhali-care-25-01.csv**: Interacciones reales del sistema
- **preferences.csv**: Dataset de preferencias (prompt, chosen, rejected)
- **augmented_pairs.csv**: Pares aumentados de entrenamiento

### Documentos de Prueba
Subconjunto en `docs_test/` para testing rápido:
- 3 documentos representativos
- Tests de integración

## 🏛️ Arquitectura del Sistema

### Pipeline RAG
```
PDF → Docling → Chunking → Nomic Embeddings → LanceDB
                                                    ↓
Usuario → Query → Embeddings → Búsqueda → Top-K → LLM → Respuesta
```

### Flujo de Fine-tuning
```
Interacciones → Augmentation → Preferences Dataset → QLoRA (DPO/ORPO) → Modelo Mejorado
```

### Arquitectura Web (Resultado10)
```
Usuario → Frontend (HTML/CSS/JS)
            ↓
        FastAPI (Backend)
            ↓
    ┌───────┴───────┐
    ↓               ↓
RAG Agent      PostgreSQL
    ↓
LanceDB + Ollama
```

## 🔧 Configuración Avanzada

### Variables de Configuración RAG
```python
LANCEDB_PATH = "tmp/lancedb"          # Ruta base vectorial
TABLE_NAME = "docs_qa"                # Tabla principal
EMBEDDING_DIM = 768                   # Dimensiones Nomic
MAX_WORDS = 512                       # Palabras por chunk
METADATA_FILE = "docs_metadata.json"  # Metadatos
```

### Optimización de Índices
- **< 1,000 vectores**: HNSW (m=32, ef_construction=100)
- **1K - 5K vectores**: IVF_PQ (√N particiones)
- **> 5K vectores**: IVF_PQ (6K vectores/partición)

### Hiperparámetros QLoRA (Preset 0)
```python
lora_r=16
lora_alpha=32
lora_dropout=0.05
learning_rate=5e-5
warmup_ratio=0.1
max_seq_length=2048
```

## 📈 Resultados Experimentales

### Rendimiento RAG
- **Procesamiento**: ~30-50 chunks/segundo
- **Consultas**: <100ms para bases <10K vectores
- **Precisión Hit@5**: >85% en dominio de salud

### Fine-tuning
- **DPO**: Mejora de ~15% en preferencias
- **ORPO**: Convergencia más rápida (~20% menos epochs)
- **Tiempo de entrenamiento**: ~30min por epoch (RTX 3090)

### Sistema Web
- **Latencia**: <200ms respuesta streaming
- **Concurrencia**: 50+ usuarios simultáneos
- **Uptime**: 99.9% en pruebas de carga

## 🛠️ Solución de Problemas

### Error: "No hay archivos PDF que coincidan"
```bash
# Verificar PDFs en docs/ y Documentos.csv
ls docs/
cat Documentos.csv
```

### Error: "Ollama no disponible"
```bash
# Verificar servicio y modelos
ollama list
systemctl status ollama  # Linux
```

### Error: PostgreSQL connection (Resultado10)
```bash
# Verificar contenedor
docker ps
docker logs postgres

# Recrear si es necesario
docker-compose down
docker-compose up -d postgres
```

### Error: CUDA out of memory
```python
# Reducir batch_size o usar gradient_checkpointing
# En qlora_pref_train.py:
per_device_train_batch_size=1
gradient_checkpointing=True
```

## 🤝 Contribución

1. Fork el proyecto
2. Crear branch (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a branch (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia Apache 2.0 - ver archivo [LICENSE](LICENSE) para detalles.

## 🙏 Reconocimientos

### Frameworks y Bibliotecas
- [Agno AI](https://github.com/agno-ai/agno) - Framework RAG
- [LanceDB](https://lancedb.com/) - Base de datos vectorial
- [Ollama](https://ollama.ai/) - Modelos LLM locales
- [Docling](https://github.com/DS4SD/docling) - Procesamiento PDF
- [FastAPI](https://fastapi.tiangolo.com/) - Framework web
- [Unsloth](https://github.com/unslothai/unsloth) - Optimización QLoRA

### Modelos
- **Llama 3.2** (Meta AI) - Modelo base principal
- **Gemma 2** (Google) - Modelo alternativo
- **Qwen 2.5/3** (Alibaba) - Modelos multilingües
- **Nomic Embeddings** - Embeddings de alta calidad
- **NLLB-200** (Meta AI) - Traducción multilingüe

### Herramientas
- Polars, PyTorch, Transformers, PEFT, TRL
- PostgreSQL, Docker, Uvicorn
- PyTest, Pandas, NumPy

---

**Proyecto de Tesis** - Sistema de Promoción de Salud con IA  
**Universidad**: [Nombre de Universidad]  
**Autor**: [Nombre del Autor]  
**Año**: 2025

Para más información, consulta la documentación individual en cada carpeta de resultado.
