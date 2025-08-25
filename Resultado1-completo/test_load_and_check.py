#Ejecutar con pytest test_load_and_check.py

import os
import shutil
import pytest
import polars as pl
from pathlib import Path
from datetime import datetime

from load_and_check import (
    ensure_pdf_dir_exists,
    connect_database,
    prepare_table,
    init_converter_and_embedder,
    process_pdfs,
    insert_records,
    load_documentos_info,
    save_metadata_json,
    PdfChunk,
    LANCEDB_PATH,
    TABLE_NAME,
    EMBEDDING_DIM,
    METADATA_FILE,
)

# Ruta a PDFs de prueba
TEST_PDF_DIR = os.path.join(os.path.dirname(__file__), "docs_test")

def cleanup_session_start():
    """Limpia SOLO archivos temporales al inicio de la sesión de tests"""
    print("\n🧹 Limpiando archivos temporales previos...")
    
    # Limpiar base de datos temporal
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
        print("   ✓ Base de datos temporal eliminada")
    
    # Limpiar archivo de metadatos
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
        print("   ✓ Archivo de metadatos eliminado")
    
    # Mostrar archivos de resultados existentes
    result_files = [f for f in os.listdir(".") if f.startswith("test_results_") and f.endswith(".txt")]
    if result_files:
        print(f"   📄 Archivos de resultados existentes: {len(result_files)}")
        for f in result_files:
            print(f"      - {f}")
    else:
        print("   📄 No hay archivos de resultados previos")

# Ejecutar limpieza al importar el módulo
cleanup_session_start()

def cleanup_test_files():
    """Limpia archivos temporales de test (NO los archivos de resultados permanentes)"""
    # Limpiar base de datos de test
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    # Limpiar archivo de metadatos
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)

def run_word_count_test(selected_pdfs: list[str], test_name: str, output_file: str):
    """
    Ejecuta test de conteo de palabras para una lista específica de PDFs.
    Retorna información detallada del test.
    
    Args:
        selected_pdfs: Lista de PDFs a procesar
        test_name: Nombre descriptivo del test
        output_file: Archivo donde guardar los resultados (permanente)
    """
    print(f"\n=== INICIANDO {test_name} ===")
    
    # Limpiar archivos temporales previos
    cleanup_test_files()
    
    # Crear base de datos temporal
    db_path = "tmp/test_lancedb"
    os.makedirs(db_path, exist_ok=True)
    db = connect_database(db_path)
    table = prepare_table(db, TABLE_NAME, PdfChunk)
    converter, embedder = init_converter_and_embedder("nomic-embed-text-v2", EMBEDDING_DIM)
    
    # Verificar carpeta de test
    ensure_pdf_dir_exists(TEST_PDF_DIR)
    all_pdf_files = [f for f in os.listdir(TEST_PDF_DIR) if f.lower().endswith(".pdf")]
    
    # Filtrar solo los PDFs seleccionados que existen
    test_pdfs = [pdf for pdf in selected_pdfs if pdf in all_pdf_files]
    if not test_pdfs:
        raise ValueError(f"Ninguno de los PDFs seleccionados existe en {TEST_PDF_DIR}")
    
    print(f"PDFs a procesar: {test_pdfs}")
    
    # Cargar información de documentos
    docs_info = load_documentos_info()
    
    # Procesar PDFs de prueba
    print("\n--- PROCESAMIENTO DE PDFs ---")
    records, doc_word_counts, metadata = process_pdfs(test_pdfs, TEST_PDF_DIR, converter, embedder, EMBEDDING_DIM, docs_info)
    
    # Mostrar información pre-inserción
    print("\n--- CONTEO PRE-INSERCIÓN ---")
    for pdf_name, word_count in doc_word_counts.items():
        titulo = ""
        if metadata and pdf_name in metadata.get("documents", {}):
            titulo = metadata["documents"][pdf_name].get("titulo", "")
        print(f"Archivo: {pdf_name}")
        print(f"  Título: {titulo}")
        print(f"  Palabras antes de LanceDB: {word_count}")
        print()
    
    # Guardar metadatos e insertar registros
    save_metadata_json(metadata)
    insert_records(table, records)
    
    # Cargar tabla y verificar conteos
    df = pl.from_arrow(table.to_arrow())
    
    print("--- VERIFICACIÓN POST-INSERCIÓN ---")
    test_results = []
    
    for pdf_name, original_count in doc_word_counts.items():
        # Filtrar por PDF específico y concatenar textos manteniendo orden de chunks
        sub = df.filter(pl.col("pdf_name") == pdf_name)
        texts = sub.sort("chunk_index").select("text").to_series().to_list()
        concatenated = " ".join(texts)
        query_count = len(concatenated.split())
        
        # Obtener título de metadatos
        titulo = ""
        if metadata and pdf_name in metadata.get("documents", {}):
            titulo = metadata["documents"][pdf_name].get("titulo", "")
        
        print(f"Archivo: {pdf_name}")
        print(f"  Título: {titulo}")
        print(f"  Palabras antes de LanceDB: {original_count}")
        print(f"  Palabras después de consulta: {query_count}")
        print(f"  ✓ Match: {original_count == query_count}")
        print()
        
        test_results.append({
            "pdf_name": pdf_name,
            "titulo": titulo,
            "original_count": original_count,
            "query_count": query_count,
            "match": original_count == query_count
        })
        
        # Prueba de consistencia
        assert query_count == original_count, (
            f"❌ FALLO en {pdf_name} (Título: {titulo}): "
            f"Pre-LanceDB={original_count}, Post-consulta={query_count}"
        )
    
    # Escribir resultados detallados a archivo PERMANENTE
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== RESULTADOS {test_name} ===\n")
        f.write(f"Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PDFs procesados: {len(test_results)}\n\n")
        
        for result in test_results:
            f.write(f"Archivo: {result['pdf_name']}\n")
            f.write(f"Título: {result['titulo']}\n")
            f.write(f"Palabras pre-LanceDB: {result['original_count']}\n")
            f.write(f"Palabras post-consulta: {result['query_count']}\n")
            f.write(f"Match exitoso: {'✓' if result['match'] else '❌'}\n")
            f.write("-" * 50 + "\n")
        
        f.write(f"\n=== RESUMEN ===\n")
        f.write(f"Total documentos: {len(test_results)}\n")
        f.write(f"Tests exitosos: {sum(1 for r in test_results if r['match'])}\n")
        f.write(f"Tests fallidos: {sum(1 for r in test_results if not r['match'])}\n")
    
    print(f"✅ {test_name} COMPLETADO EXITOSAMENTE")
    print(f"Todos los {len(test_results)} documentos pasaron la prueba de consistencia")
    print(f"📄 Resultados guardados en: {output_file}")
    
    # Limpiar archivos temporales al final (NO el archivo de resultados)
    cleanup_test_files()
    
    return test_results

def test_word_count_consistency_1_document():
    """
    Test con 1 documento: Verifica consistencia de conteo de palabras.
    """
    # Usar el primer PDF disponible
    all_pdfs = [f for f in os.listdir(TEST_PDF_DIR) if f.lower().endswith(".pdf")]
    if not all_pdfs:
        pytest.skip("No hay PDFs de prueba disponibles")
    
    selected_pdfs = [all_pdfs[0]]
    output_file = "test_results_1_documento.txt"
    results = run_word_count_test(selected_pdfs, "TEST 1 DOCUMENTO", output_file)
    assert len(results) == 1
    assert all(r["match"] for r in results)

def test_word_count_consistency_2_documents():
    """
    Test con 2 documentos: Verifica consistencia respetando orden de inserción.
    """
    all_pdfs = [f for f in os.listdir(TEST_PDF_DIR) if f.lower().endswith(".pdf")]
    if len(all_pdfs) < 2:
        pytest.skip("Se necesitan al menos 2 PDFs de prueba")
    
    selected_pdfs = all_pdfs[:2]
    output_file = "test_results_2_documentos.txt"
    results = run_word_count_test(selected_pdfs, "TEST 2 DOCUMENTOS", output_file)
    assert len(results) == 2
    assert all(r["match"] for r in results)

def test_word_count_consistency_3_documents():
    """
    Test con 3 documentos: Verifica consistencia respetando orden de inserción.
    """
    all_pdfs = [f for f in os.listdir(TEST_PDF_DIR) if f.lower().endswith(".pdf")]
    if len(all_pdfs) < 3:
        pytest.skip("Se necesitan al menos 3 PDFs de prueba")
    
    selected_pdfs = all_pdfs[:3]
    output_file = "test_results_3_documentos.txt"
    results = run_word_count_test(selected_pdfs, "TEST 3 DOCUMENTOS", output_file)
    assert len(results) == 3
    assert all(r["match"] for r in results)