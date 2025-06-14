import os
import re
import unicodedata
import random
import sys
from math import ceil, sqrt

import numpy as np
import pyarrow as pa
import polars as pl

import lancedb
from lancedb.pydantic import LanceModel, Vector
from docling.document_converter import DocumentConverter
from agno.embedder.ollama import OllamaEmbedder


# Constantes de configuración
LANCEDB_PATH = "tmp/lancedb"
TABLE_NAME = "docs_qa"
EMBEDDING_DIM = 768  # Dimensiones para embedder nomic-embed-text
PDF_DIR = os.path.join(os.path.dirname(__file__), "docs")
MAX_WORDS = 512

# Pydantic schema for PDF chunks
class PdfChunk(LanceModel):
    pdf_name: str
    chunk_index: int
    text: str
    vector: Vector(EMBEDDING_DIM)

def ensure_pdf_dir_exists(pdf_dir: str):
    # Verifica que el directorio exista
    if not os.path.exists(pdf_dir):
        raise FileNotFoundError(
            f"La carpeta '{pdf_dir}' no existe. Crea una carpeta 'docs' y coloca tus PDF ahi."
        )
    return pdf_dir

def connect_database(path: str):
    #Conectar (o crear) la base LanceDB
    os.makedirs(path, exist_ok=True)
    return lancedb.connect(path)

def prepare_table(db, table_name: str, schema):
    # Si ya existiera la tabla, se elimina para recrear limpia
    if table_name in db.table_names():
        db.drop_table(table_name)
    table = db.create_table(table_name, schema=schema)
    print(f"Tabla preparada: {table_name}")
    return table

def init_converter_and_embedder(embedder_id: str ,embedding_dim: int):
    # Docling para extraer texto de PDF
    converter = DocumentConverter()
    # OllamaEmbedder con el modelo embedder
    embedder = OllamaEmbedder(id=embedder_id, dimensions=embedding_dim)
    print(f"Ollama embedder {embedder_id} cargado exitosamente a {embedding_dim} dimensiones")
    return converter, embedder

def list_pdf_files(pdf_dir: str):
    files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not files:
        raise FileNotFoundError("No hay archivos .pdf. encontrados en el archivo")
    return files

def select_pdfs(files: list[str], selection: str=None) -> list[str]:
    if selection is not None:
        print("Selecciona 1 o mas PDF (indices separados por comas) o '.' para todos:")
        for idx, fn in enumerate(files, 1):
            print(f"  {idx}. {fn}")

    selection = input("Indices (por ejemplo 1, 3 or .): ").strip()
    if selection == '.':
        return files

    indices = [int(i) - 1 for i in selection.split(",") if i.strip().isdigit()]
    chosen = [files[i] for i in indices if 0 <= i < len(files)]
    if not chosen:
        raise ValueError("No hay PDF validos seleccionados.")
    print(f"Processing {len(chosen)} PDF(s): {chosen}")
    return chosen

# Tabla de mapeo
UNICODE_REPLACEMENTS = {
    "/uniFB00": "ff",   # ﬀ
    "/uniFB01": "fi",   # ﬁ
    "/uniFB02": "fl",   # ﬂ
    "/uniFB03": "ffi",  # ﬃ
    "/uniFB04": "ffl",  # ﬄ
    "/uni2013": "-",    # en dash
    "/uni2014": "-",    # em dash
    "/uni2018": "'",    # open single quote
    "/uni2019": "'",    # close single quote
    "/uni201C": '"',    # open double quote
    "/uni201D": '"',    # close double quote
    # Añadir si hay más
}

def replace_unicode_chars(text: str) -> str:
    for key, val in UNICODE_REPLACEMENTS.items():
        text = re.sub(rf"\s*{re.escape(key)}\s*", val, text)
    return text

def merge_unbalanced_parentheses(paragraphs: list[str]) -> list[str]:
    merged, buf, balance = [], None, 0

    for p in paragraphs:
        opens = p.count("(")
        closes = p.count(")")
        if buf is None:
            # inicio de posible párrafo partido
            if opens > closes:
                buf = p
                balance = opens - closes
            else:
                merged.append(p)
        else:
            # seguimos juntando
            buf += " " + p
            balance += opens - closes
            if balance <= 0:
                # cerramos el buffer
                merged.append(buf)
                buf = None
                balance = 0

    # si quedó algo sin cerrar, lo agregamos
    if buf is not None:
        merged.append(buf)

    return merged


def fix_broken_urls(text: str) -> str:
    """
    Detecta subcadenas que empiezan por http:// o https://
    y contienen espacios, y los elimina solo dentro de la URL.
    """
    # Captura desde http:// o https:// hasta el primer espacio que no forme parte de la URL
    # Permite letras, dígitos, punto, slash, ?, =, &, %, -, #, :, _
    url_re = re.compile(r'(https?://[\w\.\-/%=&\?#:_\s]+)')
    def _join_spaces(match):
        url = match.group(1)
        # Borra únicamente los espacios dentro de la URL
        return url.replace(" ", "")
    return url_re.sub(_join_spaces, text)

def split_references_section(paragraphs: list[str]) -> list[str]:
    out, buffer, in_refs = [], [], False

    # 1) Separa pre-referencias vs referencias
    for p in paragraphs:
        if p.strip().startswith("## References"):
            out.append(p.strip())
            in_refs = True
            continue
        if in_refs:
            buffer.append(p.strip())
        else:
            out.append(p)

    if in_refs and buffer:
        # Unimos todo sin saltos y colapsamos espacios
        blob = " ".join(buffer)
        blob = re.sub(r"\s+", " ", blob).strip()
        # Recomposicion de URL partidas
        blob = fix_broken_urls(blob)

        # Split inicial por '- <número>.'
        refs = re.split(r"\s*-\s*(?=\d+\.)", blob)

        # Refinamiento: unión de fragmentos cortados e ignora trozos < 3 palabras
        cleaned, i = [], 0
        while i < len(refs):
            ref = refs[i].strip()

            # Si termina en dígitos sin punto y el siguiente empieza con 'm.' → juntamos
            if re.search(r"\d+$", ref) and not ref.endswith(".") and i + 1 < len(refs):
                nxt = refs[i+1].strip()
                if re.match(r"^\d+\.", nxt):
                    ref = f"{ref} {nxt}"
                    i += 1

            words = ref.split()
            # Fragmentos muy cortos o solo número → fusionar con anterior
            if len(words) < 3 or re.match(r"^\d+\.?$", ref):
                if cleaned:
                    cleaned[-1] += " " + ref
                else:
                    # si no hay anterior, fusiona con el siguiente
                    if i + 1 < len(refs):
                        refs[i+1] = f"{ref} {refs[i+1].strip()}"
                i += 1
                continue

            cleaned.append(ref)
            i += 1

        # Volvemos a añadir las referencias limpias
        for ref in cleaned:
            out.append(ref)

    return out

def preprocess_text(raw: str) -> str:
    # Limpieza de ligaduras y comillas
    t = replace_unicode_chars(raw)

    # Unir saltos simples, colapsar espacios
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
    t = re.sub(r"[ \t]+", " ", t)

    # Split inicial por párrafos
    paras = re.split(r"\n\s*\n", t)

    # Fusionar paréntesis desbalanceados
    paras = merge_unbalanced_parentheses(paras)

    # Procesar referencias como un bloque aparte
    paras = split_references_section(paras)

    # 6) Reconstruir texto preprocesado
    return "\n\n".join(paras)

# Función para dividir texto en chunks
def naive_chunkify(text: str, max_tokens: int = MAX_WORDS) -> list[str]:
    """
    Divide 'text' en fragmentos aproximados de 'max_tokens' palabras.
    """
    words = text.split()
    return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def alternate_chunk_flow(paragraph: str) -> list[str]:
    """
    Flujo alterno para procesar párrafos demasiado largos.
    Actualmente, divide el párrafo en oraciones. En el futuro
    aquí se podrá llamar a funciones de síntesis o resumen.
    """
    sentences = re.split(r'(?<=[\.\?\!])\s+', paragraph)
    return [s.strip() for s in sentences if s.strip()]


def chunkify_by_paragraphs(text: str) -> list[str]:
    """
    Divide el texto en chunks basados en párrafos (separados por una línea vacía).
    Si un párrafo es muy largo (más de MAX_WORDS), se subdivide en fragmentos más pequeños.
    """

    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for para in raw_paragraphs:
        words = para.split()
        if len(words) <= MAX_WORDS:
            # Si el párrafo cabe entero, lo agregamos directamente
            chunks.append(para)
        else:
            # Párrafo demasiado largo: invocar el flujo alterno
            chunks.extend(alternate_chunk_flow(para))
    return chunks

def process_pdfs(
    pdf_names: list[str], pdf_dir: str, converter, embedder, embedding_dim: int
) -> tuple[list[dict], dict[str,int]]:
    """
    Procesa cada PDF, genera registros chunk + embeddings,
    y devuelve además un mapping pdf_name -> total_words (pre-chunk).
    """    
    records = []
    doc_word_counts: dict[str,int] = {}    

    for pdf_name in pdf_names:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        print(f"\nProcesando texto del PDF: {pdf_name}")

        try:
            # convert() retorna un objeto con .document
            result = converter.convert(pdf_path)
            doc = result.document
        except Exception as e:
            print(f"Error convirtiendo '{pdf_name}': {e}")
            continue

        # Extraemos todo el texto del PDF (export_to_text devuelve un string grande)
        try:
            raw_text = doc.export_to_text()
            full_text = preprocess_text(raw_text)
            total_words = len(full_text.split())
            doc_word_counts[pdf_name] = total_words
        except Exception as e:
            print(f"Error en extraccion de texto de '{pdf_name}': {e}")
            continue

        if total_words == 0:
            print(f"PDF '{pdf_name}' devolvió un texto vacío.")
            continue

        chunks = chunkify_by_paragraphs(full_text)
        print(f"Se obtuvieron {len(chunks)} chunks de texto para '{pdf_name}' (palabras: {total_words}).")

        for idx, chunk in enumerate(chunks):
            try:
                emb = embedder.get_embedding(chunk)
            except Exception as e:
                print(f"Error embebiendo chunk {idx}: {e}")
                emb = [0.0] * embedding_dim

            emb_np = np.array(emb, dtype=np.float32)
            if emb_np.shape[0] != embedding_dim or np.any(np.isnan(emb_np)):
                emb_np = np.zeros((embedding_dim,), dtype=np.float32)

            records.append({
                "pdf_name": pdf_name,
                "chunk_index": idx,
                "text": chunk,
                "vector": emb_np.tolist(),
            })

    return records, doc_word_counts

def insert_records(table, records: list[dict]):
    if not records:
        print("\nNo hay registros generados. El pipeline termina.")
        sys.exit(0)
    print(f"\nInsertando {len(records)} registros en '{TABLE_NAME}'...")
    table.add(records, on_bad_vectors="fill", fill_value=0.0)
    print("Registros insertados correctamente.")
    return len(records)

def create_vector_index(table, total_vectors: int, embedding_dim: int):
    print("\nCreando indice vectorial en columna 'vector' ...")
    if total_vectors < 1000:
        # Caso 1: Dataset pequeño: HNSW
        # Para D >= 768 conviene m = 32–48; elegimos m = 32
        # construccion ef >= max(100, 2*m) para buena calidad sin sobrecargar excesivamente
        m = 32
        ef = max(100, 2 * m)
        print(f"Dataset pequeño ({total_vectors} vectores) → usando HNSW con m={m}, ef_construction={ef}")  # 
        table.create_index(
            index_type="IVF_HNSW_SQ",
            metric="cosine",
            vector_column_name="vector",
            replace=True,
            m=m,
            ef_construction=ef,
        )
    elif total_vectors < 5000:
        # Caso 2: Dataset medio: IVF-PQ con √n particiones
        # Regla de oro: num_partitions ≈ √N 
        num_partitions = max(2, int(sqrt(total_vectors)))  # al menos 2 clusters
        # numero de subvectores debe dividir D tal que D/num_sub_vectors sea múltiplo de 8
        num_sub_vectors = embedding_dim // 8
        print(f"Dataset medio ({total_vectors} vectores) → usando IVF_PQ con {num_partitions} particiones y {num_sub_vectors} sub-vectores")
        table.create_index(
            index_type="IVF_PQ",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
            replace=True,
            num_bits=8,
            max_iterations=50,
            sample_rate=total_vectors if total_vectors < 1024 else 1024,
        )
    else:
        # Caso 3: Dataset grande (n >= 5000): IVF-PQ con ~6000 vectores/partición
        # Regla de oro: cada partición debería contener 4k–8k vectores; elegimos ~6000
        target = 6000
        num_partitions = max(1, ceil(total_vectors / target))
        num_sub_vectors = embedding_dim // 8 # por ejemplo 768/8 = 96
        print(f"Dataset grande ({total_vectors} vectores) → usando IVF_PQ con {num_partitions} particiones y {num_sub_vectors} sub-vectores")
        table.create_index(
            index_type="IVF_PQ",
            metric="cosine",
            num_partitions=num_partitions,
            num_sub_vectors=num_sub_vectors,
            vector_column_name="vector",
            replace=True,
            num_bits=8,
            max_iterations=50,
            # sample_rate fijo en 1024, suficiente para entrenar PQ en miles de vectores
            sample_rate=1024,
        )
    print("Indice vectorial creado.")

def preview_records(records: list[dict], sample_size: int = 3):
    # Mostrar vista previa de lo ingresado
    if records:
        print("\n\nVista previa de embeddings insertados:")
        # Tomamos hasta 3 registros al azar
        sample = random.sample(records, k=min(sample_size, len(records)))
        for rec in sample:
            txt = rec["text"]
            vec = rec["vector"]
            print(f"• PDF: {rec['pdf_name']}  (chunk_index={rec['chunk_index']})")
            print(f"  Texto (primeros 50 caracteres): {txt[:50].replace(chr(10), ' ')}...")
            print(f"  Vector (primeros 10 valores): {vec[:10]}...")
            print("---")
    else:
        print("No registros para mostrar.")

def create_fts_index(table):
    print("\nIntentando crear índice FTS en la columna 'text'...")
    schema = table.schema
    cols = [f.name for f in schema]
    types = {f.name: f.type for f in schema}
    if "text" not in cols:
        print("La columna 'text' no existe en el esquema.")
    elif types["text"] != pa.string():
        print(f"La columna 'text' tiene tipo inesperado: {types['text']}")
    else:
        try:
            table.create_fts_index(
                field_names=["text"],
                replace=True,
                tokenizer_name="default",
                base_tokenizer="simple",
                language="English",
                lower_case=True,
                stem=True,
                remove_stop_words=True,
                ascii_folding=True,
            )
            print("Índice FTS creado exitosamente.")
        except Exception as e:
            print(f"Error al crear el índice FTS: {e}")

def query_documents(table):
    """
    Menú de consulta usando Polars en vez de Pandas.
    """
    # 1. Cargamos todo en un DataFrame de Polars
    df = pl.from_arrow(table.to_arrow())

    # 2. Extraemos lista de documentos únicos
    docs = df.select("pdf_name").unique().to_series().to_list()

    # 3. Menú de selección
    print("\nSelecciona el/los documento(s) a consultar:")
    for i, name in enumerate(docs, 1):
        print(f"  {i}. {name}")
    sel = input("Índices (ej. 1,3): ").strip()
    idxs = [int(x)-1 for x in sel.split(",") if x.strip().isdigit()]
    chosen = [docs[i] for i in idxs if 0 <= i < len(docs)]
    if not chosen:
        raise ValueError("No se seleccionó ningún documento válido.")

    # 4. Para cada documento elegido, calculamos métricas y opcionalmente mostramos texto
    for pdf in chosen:
        sub = df.filter(pl.col("pdf_name") == pdf)
        n_chunks = sub.height
        # Contamos palabras sumando longitud de split() de cada texto
        total_words = (
            sub
            .with_columns(pl.col("text")
                         .str.split(" ")
                         .list.len()
                         .alias("word_count"))
            .select(pl.col("word_count"))
            .sum()
            .item()
        )
        print(f"\nDocumento: {pdf}")
        print(f" • Chunks almacenados: {n_chunks}")
        print(f" • Palabras totales (suma de chunks): {total_words}")

        if input("¿Mostrar texto completo concatenado? (y/N): ").lower() == "y":
            # Concatenamos manteniendo orden de chunk_index
            full = (
                sub
                .sort("chunk_index")
                .select("text")
                .to_series()
                .to_list()
            )
            print("\n--- Inicio texto completo ---\n")
            print("\n\n".join(full))
            print("\n--- Fin texto completo ---\n")

def main():
    pdf_dir = ensure_pdf_dir_exists(PDF_DIR)
    db = connect_database(LANCEDB_PATH)
    table = prepare_table(db, TABLE_NAME, PdfChunk)
    converter, embedder = init_converter_and_embedder("nomic-embed-text-v2",EMBEDDING_DIM)
    pdf_files = list_pdf_files(pdf_dir)
    selected_pdfs = select_pdfs(pdf_files)
    records, doc_word_counts  = process_pdfs(selected_pdfs, pdf_dir, converter, embedder, EMBEDDING_DIM)
    total = insert_records(table, records)
    create_vector_index(table, total, EMBEDDING_DIM)
    preview_records(records)
    create_fts_index(table)
    print(f"\nBase vectorial LanceDB lista en '{LANCEDB_PATH}' con {total} registros.")
    query_documents(table)

if __name__ == "__main__":
    main()
