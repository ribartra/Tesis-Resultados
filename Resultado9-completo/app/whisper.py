import os
import torch
import uuid
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment

# Configurar dispositivo y tipo de tensor según la disponibilidad de GPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Definir el identificador del modelo (en este ejemplo se usa "openai/whisper-large-v3")
model_id = "openai/whisper-large-v3"

# Cargar modelo y procesador (tokenizer, feature extractor)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

# Crear el pipeline para transcripción con Whisper
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,   # Esto devuelve los timestamps (por si es necesario)
    chunk_length_s=15,        # Divide el audio en segmentos de 15 segundos para evitar problemas en audios largos
    batch_size=16,
)

def convert_to_wav(audio_path: str) -> str:
    """
    Convierte el archivo a formato WAV si la extensión no es .mp3 o .wav.
    Devuelve la ruta del archivo convertido o la misma si ya es .mp3 o .wav.
    """
    file_name, file_ext = os.path.splitext(audio_path)
    if file_ext.lower() not in ['.mp3', '.wav','.webm']:
        print(f"Convirtiendo {audio_path} a formato WAV...")
        audio = AudioSegment.from_file(audio_path)
        new_audio_path = f"{file_name}_{uuid.uuid4().hex}.wav"
        audio.export(new_audio_path, format="wav")
        print(f"Archivo guardado como {new_audio_path}")
        return new_audio_path
    return audio_path

def transcribe(audio_path: str) -> str:
    """
    Realiza la transcripción del archivo de audio usando el pipeline de Whisper.
    Convierte a WAV si es necesario y retorna el texto transcrito.
    """
    # Convertir el archivo a WAV si no lo está
    audio_file = convert_to_wav(audio_path)
    result = whisper_pipeline(audio_file)
    return result.get('text', '')
