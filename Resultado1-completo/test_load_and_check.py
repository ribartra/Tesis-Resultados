#Ejecutar con pytest test_load_and_check.py

import os
import shutil
import pytest
import polars as pl
from pathlib import Path

from load_and_check import (
    ensure_pdf_dir_exists,
    connect_database,
    prepare_table,
    init_converter_and_embedder,
    process_pdfs,
    insert_records,
    PdfChunk,
    LANCEDB_PATH,
    TABLE_NAME,
    EMBEDDING_DIM,
)

# Ruta a PDFs de prueba
TEST_PDF_DIR = os.path.join(os.path.dirname(__file__), "docs_test")

@pytest.fixture(scope="module")
def lancedb_env(tmp_path_factory):
    # Crear base de datos de prueba en carpeta temporal
    db_path = str(tmp_path_factory.mktemp("test_lancedb"))
    db = connect_database(db_path)
    table = prepare_table(db, TABLE_NAME, PdfChunk)
    converter, embedder = init_converter_and_embedder("nomic-embed-text-v2", EMBEDDING_DIM)
    
    # Verificar carpeta de test
    ensure_pdf_dir_exists(TEST_PDF_DIR)
    pdf_files = [f for f in os.listdir(TEST_PDF_DIR) if f.lower().endswith(".pdf")]

    # Procesar PDFs de prueba
    records, doc_word_counts = process_pdfs(pdf_files, TEST_PDF_DIR, converter, embedder, EMBEDDING_DIM)
    insert_records(table, records)

    yield table, records, doc_word_counts

    # Limpiar base de datos al finalizar
    shutil.rmtree(db_path)

def test_word_counts_consistency_and_output(lancedb_env, tmp_path):
    """
    Comprueba que el conteo de palabras antes y después de cargar a la BD coincide al 100%
    y genera un archivo 'test_results.txt' con los resultados.
    """
    table, records, doc_word_counts = lancedb_env

    # Cargar toda la tabla en un DataFrame de Polars
    df = pl.from_arrow(table.to_arrow())

    # Ruta de salida
    out_file = Path("test_results.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for pdf_name, original_count in doc_word_counts.items():
            sub = df.filter(pl.col("pdf_name") == pdf_name)
            # Concatenar todos los chunks y contar palabras
            texts = sub.sort("chunk_index").select("text").to_series().to_list()
            concatenated = " ".join(texts)
            query_count = len(concatenated.split())

            # Escribir resultado al archivo
            f.write(f"{pdf_name}: original={original_count}, query={query_count}\n")
            # Prueba de consistencia
            assert query_count == original_count, (
                f"Word count mismatch for {pdf_name}: "
                f"original={original_count}, query={query_count}"
            )

    # Verificar que el archivo fue creado y contiene líneas
    content = out_file.read_text(encoding="utf-8")
    assert content.strip(), "El archivo 'test_results.txt' está vacío"
    # Opcional: imprimir para debug
    print(content)
