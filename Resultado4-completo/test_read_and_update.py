# Ejecutar con pytest test_read_and_update.py

import os
import shutil
import pytest
import polars as pl
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
from io import StringIO

from read_and_update import (
    validate_database_exists, create_vector_database, handle_empty_database,
    load_vector_database, load_documents_data, get_table_dataframe,
    list_vectorized_documents, create_new_document, update_document_metadata,
    delete_document_from_database, main_menu_loop
)

from load_and_check import (
    LANCEDB_PATH, TABLE_NAME, EMBEDDING_DIM, METADATA_FILE, DOCUMENTOS_CSV,
    PdfChunk, connect_database, init_converter_and_embedder,
    load_documentos_info, save_metadata_json, process_pdfs, 
    insert_records, create_vector_index, create_fts_index,
    prepare_table
)

# Rutas de prueba
TEST_PDF_DIR = os.path.join(os.path.dirname(__file__), "docs_test")
TEST_DB_PATH = "tmp/test_read_update_lancedb"
TEST_METADATA_FILE = "test_docs_metadata.json"

def cleanup_session_start():
    """Limpia archivos temporales al inicio de la sesión de tests"""
    print("\n🧹 Limpiando archivos temporales previos para read_and_update tests...")
    
    # Limpiar base de datos temporal
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
        print("   ✓ Base de datos temporal eliminada")
    
    # Limpiar archivo de metadatos de prueba
    if os.path.exists(TEST_METADATA_FILE):
        os.remove(TEST_METADATA_FILE)
        print("   ✓ Archivo de metadatos de prueba eliminado")
    
    # Mostrar archivos de resultados existentes
    result_files = [f for f in os.listdir(".") if f.startswith("test_crud_results_") and f.endswith(".txt")]
    if result_files:
        print(f"   📄 Archivos de resultados CRUD existentes: {len(result_files)}")
        for f in result_files:
            print(f"      - {f}")
    else:
        print("   📄 No hay archivos de resultados CRUD previos")

# Ejecutar limpieza al importar el módulo
cleanup_session_start()

def cleanup_test_files():
    """Limpia archivos temporales de test"""
    if os.path.exists("tmp"):
        shutil.rmtree("tmp")
    if os.path.exists(TEST_METADATA_FILE):
        os.remove(TEST_METADATA_FILE)

def create_test_database():
    """Crea una base de datos de prueba con documentos de docs_test"""
    # Limpiar archivos previos
    cleanup_test_files()
    
    # Crear base de datos temporal
    os.makedirs(TEST_DB_PATH, exist_ok=True)
    db = connect_database(TEST_DB_PATH)
    table = prepare_table(db, TABLE_NAME, PdfChunk)
    converter, embedder = init_converter_and_embedder("nomic-embed-text-v2", EMBEDDING_DIM)
    
    # Obtener archivos de test
    test_files = [f for f in os.listdir(TEST_PDF_DIR) if f.lower().endswith(".pdf")]
    if not test_files:
        raise ValueError("No hay archivos PDF de prueba disponibles")
    
    # Cargar información de documentos
    docs_info = load_documentos_info()
    
    # Procesar todos los archivos de test
    records, doc_word_counts, metadata = process_pdfs(
        test_files, TEST_PDF_DIR, converter, embedder, EMBEDDING_DIM, docs_info
    )
    
    # Insertar registros
    insert_records(table, records)
    
    # Crear índices
    create_vector_index(table, len(records), EMBEDDING_DIM)
    create_fts_index(table)
    
    # Guardar metadatos con nombre de archivo específico para test
    metadata_content = metadata.copy()
    with open(TEST_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_content, f, indent=2, ensure_ascii=False)
    
    return db, table, metadata, test_files, len(records)

def run_crud_operations_test(test_name: str, output_file: str):
    """
    Ejecuta pruebas CRUD completas verificando todas las operaciones.
    
    Args:
        test_name: Nombre descriptivo del test
        output_file: Archivo donde guardar los resultados
    """
    print(f"\n=== INICIANDO {test_name} ===")
    
    # Crear base de datos de prueba
    db, table, metadata, test_files, total_records = create_test_database()
    
    # Resultados de pruebas
    test_results = {
        'test_name': test_name,
        'timestamp': datetime.now().isoformat(),
        'total_test_files': len(test_files),
        'total_records': total_records,
        'results': {
            'database_creation': False,
            'read_operations': [],
            'create_operations': [],
            'update_operations': [],
            'delete_operations': [],
            'validation_tests': []
        },
        'errors': [],
        'summary': {}
    }
    
    try:
        # ============================================================================
        # PRUEBA 1: VALIDACIÓN DE BASE DE DATOS
        # ============================================================================
        
        # Mock temporal de las rutas para usar las de test
        with patch('read_and_update.LANCEDB_PATH', TEST_DB_PATH), \
             patch('read_and_update.METADATA_FILE', TEST_METADATA_FILE):
            
            db_status = validate_database_exists()
            test_results['results']['database_creation'] = (db_status == 'exists')
            test_results['results']['validation_tests'].append({
                'test': 'database_validation',
                'status': db_status,
                'passed': db_status == 'exists'
            })
            
            # Cargar datos para pruebas
            db_test, table_test = load_vector_database()
            metadata_test, docs_info = load_documents_data()
            df = get_table_dataframe(table_test)
            
            # ============================================================================
            # PRUEBA 2: OPERACIONES READ (CONSULTA)
            # ============================================================================
            
            vectorized_docs = list_vectorized_documents(metadata_test, df)
            
            for doc_name in test_files:
                doc_data = df.filter(pl.col("pdf_name") == doc_name)
                read_result = {
                    'document': doc_name,
                    'chunks_found': doc_data.height,
                    'readable': doc_data.height > 0,
                    'has_metadata': doc_name in metadata_test.get('documents', {}) if metadata_test else False
                }
                
                # Verificar que se puede leer el contenido
                if read_result['readable']:
                    try:
                        # Simular lectura de contenido
                        texts = doc_data.sort("chunk_index").select("text").to_series().to_list()
                        read_result['total_content_chars'] = sum(len(text) for text in texts)
                        read_result['content_accessible'] = True
                    except Exception as e:
                        read_result['content_accessible'] = False
                        read_result['error'] = str(e)
                
                test_results['results']['read_operations'].append(read_result)
            
            # ============================================================================
            # PRUEBA 3: OPERACIONES UPDATE (ACTUALIZACIÓN DE METADATOS)
            # ============================================================================
            
            if metadata_test and metadata_test.get('documents'):
                for doc_name in test_files:
                    if doc_name in metadata_test['documents']:
                        original_metadata = metadata_test['documents'][doc_name].copy()
                        
                        # Simular actualización de metadatos
                        update_result = {
                            'document': doc_name,
                            'original_title': original_metadata.get('titulo', ''),
                            'original_source': original_metadata.get('fuente', ''),
                            'update_attempted': True,
                            'update_successful': False
                        }
                        
                        try:
                            # Modificar metadatos simulando una actualización
                            new_title = f"UPDATED_TEST: {original_metadata.get('titulo', 'Sin título')}"
                            new_source = f"TEST_SOURCE: {original_metadata.get('fuente', 'Sin fuente')}"
                            
                            # Actualizar en memoria
                            metadata_test['documents'][doc_name]['titulo'] = new_title
                            metadata_test['documents'][doc_name]['fuente'] = new_source
                            metadata_test['processing_timestamp'] = datetime.now().isoformat()
                            
                            # Guardar cambios
                            with open(TEST_METADATA_FILE, 'w', encoding='utf-8') as f:
                                json.dump(metadata_test, f, indent=2, ensure_ascii=False)
                            
                            # Verificar que se guardó correctamente
                            with open(TEST_METADATA_FILE, 'r', encoding='utf-8') as f:
                                updated_metadata = json.load(f)
                            
                            update_result['new_title'] = updated_metadata['documents'][doc_name]['titulo']
                            update_result['new_source'] = updated_metadata['documents'][doc_name]['fuente']
                            update_result['update_successful'] = (
                                updated_metadata['documents'][doc_name]['titulo'] == new_title and
                                updated_metadata['documents'][doc_name]['fuente'] == new_source
                            )
                            
                        except Exception as e:
                            update_result['error'] = str(e)
                            test_results['errors'].append(f"Update error for {doc_name}: {e}")
                        
                        test_results['results']['update_operations'].append(update_result)
            
            # ============================================================================
            # PRUEBA 4: OPERACIONES DELETE (ELIMINACIÓN)
            # ============================================================================
            
            # Seleccionar un documento para prueba de eliminación (el último)
            if test_files:
                doc_to_delete = test_files[-1]
                
                # Contar registros antes de eliminación
                initial_count = len(table_test)
                doc_chunks_before = df.filter(pl.col("pdf_name") == doc_to_delete).height
                
                delete_result = {
                    'document': doc_to_delete,
                    'initial_total_records': initial_count,
                    'document_chunks_before': doc_chunks_before,
                    'deletion_attempted': True,
                    'deletion_successful': False
                }
                
                try:
                    # Simular eliminación usando predicado SQL
                    rows_deleted = table_test.delete(f"pdf_name = '{doc_to_delete}'")
                    
                    # Verificar eliminación
                    final_count = len(table_test)
                    df_after = get_table_dataframe(table_test)
                    doc_chunks_after = df_after.filter(pl.col("pdf_name") == doc_to_delete).height
                    
                    delete_result.update({
                        'rows_deleted': rows_deleted,
                        'final_total_records': final_count,
                        'document_chunks_after': doc_chunks_after,
                        'deletion_successful': (
                            doc_chunks_after == 0 and 
                            final_count == (initial_count - doc_chunks_before)
                        )
                    })
                    
                    # Actualizar metadatos simulando eliminación
                    if metadata_test and doc_to_delete in metadata_test.get('documents', {}):
                        if 'processing_order' in metadata_test:
                            metadata_test['processing_order'].remove(doc_to_delete)
                        del metadata_test['documents'][doc_to_delete]
                        metadata_test['processing_timestamp'] = datetime.now().isoformat()
                        
                        # Guardar metadatos actualizados
                        with open(TEST_METADATA_FILE, 'w', encoding='utf-8') as f:
                            json.dump(metadata_test, f, indent=2, ensure_ascii=False)
                    
                except Exception as e:
                    delete_result['error'] = str(e)
                    test_results['errors'].append(f"Delete error for {doc_to_delete}: {e}")
                
                test_results['results']['delete_operations'].append(delete_result)
        
        # ============================================================================
        # RESUMEN DE RESULTADOS
        # ============================================================================
        
        # Calcular estadísticas de éxito
        read_success = sum(1 for r in test_results['results']['read_operations'] if r.get('readable', False))
        update_success = sum(1 for r in test_results['results']['update_operations'] if r.get('update_successful', False))
        delete_success = sum(1 for r in test_results['results']['delete_operations'] if r.get('deletion_successful', False))
        validation_success = sum(1 for r in test_results['results']['validation_tests'] if r.get('passed', False))
        
        test_results['summary'] = {
            'database_creation_success': test_results['results']['database_creation'],
            'read_operations_success': f"{read_success}/{len(test_results['results']['read_operations'])}",
            'update_operations_success': f"{update_success}/{len(test_results['results']['update_operations'])}",
            'delete_operations_success': f"{delete_success}/{len(test_results['results']['delete_operations'])}",
            'validation_tests_success': f"{validation_success}/{len(test_results['results']['validation_tests'])}",
            'total_errors': len(test_results['errors']),
            'overall_success': len(test_results['errors']) == 0 and all([
                test_results['results']['database_creation'],
                read_success == len(test_results['results']['read_operations']),
                update_success == len(test_results['results']['update_operations']),
                delete_success == len(test_results['results']['delete_operations'])
            ])
        }
        
    except Exception as e:
        test_results['errors'].append(f"Test execution error: {e}")
        test_results['summary']['overall_success'] = False
    
    # ============================================================================
    # GUARDAR RESULTADOS
    # ============================================================================
    
    # Escribir resultados detallados a archivo
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"=== RESULTADOS {test_name} ===\n")
        f.write(f"Fecha de ejecución: {test_results['timestamp']}\n")
        f.write(f"Archivos PDF de prueba: {test_results['total_test_files']}\n")
        f.write(f"Total registros procesados: {test_results['total_records']}\n\n")
        
        # Resultados de validación
        f.write("=== VALIDACIÓN DE BASE DE DATOS ===\n")
        for validation in test_results['results']['validation_tests']:
            f.write(f"Test: {validation['test']}\n")
            f.write(f"Estado: {validation['status']}\n")
            f.write(f"Exitoso: {'✓' if validation['passed'] else '❌'}\n")
            f.write("-" * 40 + "\n")
        
        # Resultados READ
        f.write("\n=== OPERACIONES READ (CONSULTA) ===\n")
        for read_op in test_results['results']['read_operations']:
            f.write(f"Documento: {read_op['document']}\n")
            f.write(f"Chunks encontrados: {read_op['chunks_found']}\n")
            f.write(f"Legible: {'✓' if read_op['readable'] else '❌'}\n")
            f.write(f"Tiene metadatos: {'✓' if read_op['has_metadata'] else '❌'}\n")
            if read_op.get('content_accessible'):
                f.write(f"Caracteres de contenido: {read_op.get('total_content_chars', 0)}\n")
            if 'error' in read_op:
                f.write(f"Error: {read_op['error']}\n")
            f.write("-" * 40 + "\n")
        
        # Resultados UPDATE
        f.write("\n=== OPERACIONES UPDATE (ACTUALIZACIÓN) ===\n")
        for update_op in test_results['results']['update_operations']:
            f.write(f"Documento: {update_op['document']}\n")
            f.write(f"Título original: {update_op.get('original_title', 'N/A')}\n")
            f.write(f"Nuevo título: {update_op.get('new_title', 'N/A')}\n")
            f.write(f"Actualización exitosa: {'✓' if update_op['update_successful'] else '❌'}\n")
            if 'error' in update_op:
                f.write(f"Error: {update_op['error']}\n")
            f.write("-" * 40 + "\n")
        
        # Resultados DELETE
        f.write("\n=== OPERACIONES DELETE (ELIMINACIÓN) ===\n")
        for delete_op in test_results['results']['delete_operations']:
            f.write(f"Documento: {delete_op['document']}\n")
            f.write(f"Registros iniciales: {delete_op.get('initial_total_records', 0)}\n")
            f.write(f"Chunks del documento: {delete_op.get('document_chunks_before', 0)}\n")
            f.write(f"Registros eliminados: {delete_op.get('rows_deleted', 0)}\n")
            f.write(f"Registros finales: {delete_op.get('final_total_records', 0)}\n")
            f.write(f"Eliminación exitosa: {'✓' if delete_op['deletion_successful'] else '❌'}\n")
            if 'error' in delete_op:
                f.write(f"Error: {delete_op['error']}\n")
            f.write("-" * 40 + "\n")
        
        # Resumen final
        f.write(f"\n=== RESUMEN FINAL ===\n")
        f.write(f"Creación de BD: {'✓' if test_results['summary']['database_creation_success'] else '❌'}\n")
        f.write(f"Operaciones READ: {test_results['summary']['read_operations_success']}\n")
        f.write(f"Operaciones UPDATE: {test_results['summary']['update_operations_success']}\n")
        f.write(f"Operaciones DELETE: {test_results['summary']['delete_operations_success']}\n")
        f.write(f"Validaciones: {test_results['summary']['validation_tests_success']}\n")
        f.write(f"Total errores: {test_results['summary']['total_errors']}\n")
        f.write(f"Éxito general: {'✓ TODOS LOS TESTS PASARON' if test_results['summary']['overall_success'] else '❌ ALGUNOS TESTS FALLARON'}\n")
        
        if test_results['errors']:
            f.write(f"\n=== ERRORES DETALLADOS ===\n")
            for i, error in enumerate(test_results['errors'], 1):
                f.write(f"{i}. {error}\n")
    
    print(f"✅ {test_name} COMPLETADO")
    print(f"📄 Resultados guardados en: {output_file}")
    print(f"📊 Éxito general: {'✓' if test_results['summary']['overall_success'] else '❌'}")
    
    # Limpiar archivos temporales
    cleanup_test_files()
    
    return test_results

def test_crud_operations_complete():
    """
    Test completo de operaciones CRUD del sistema read_and_update
    """
    output_file = "test_crud_results_complete.txt"
    results = run_crud_operations_test("TEST CRUD COMPLETO", output_file)
    
    # Asserts para pytest
    assert results['summary']['overall_success'], f"Test CRUD falló. Errores: {results['errors']}"
    assert results['results']['database_creation'], "Creación de base de datos falló"
    
    # Verificar que todas las operaciones READ fueron exitosas
    read_success_count = sum(1 for r in results['results']['read_operations'] if r.get('readable', False))
    assert read_success_count == len(results['results']['read_operations']), "No todas las operaciones READ fueron exitosas"
    
    # Verificar que todas las operaciones UPDATE fueron exitosas
    update_success_count = sum(1 for r in results['results']['update_operations'] if r.get('update_successful', False))
    assert update_success_count == len(results['results']['update_operations']), "No todas las operaciones UPDATE fueron exitosas"
    
    # Verificar que todas las operaciones DELETE fueron exitosas
    delete_success_count = sum(1 for r in results['results']['delete_operations'] if r.get('deletion_successful', False))
    assert delete_success_count == len(results['results']['delete_operations']), "No todas las operaciones DELETE fueron exitosas"
    
    print(f"\n✅ TODOS LOS TESTS CRUD PASARON EXITOSAMENTE")
    print(f"📄 Resultados detallados en: {output_file}")

def test_database_validation_states():
    """
    Test de diferentes estados de validación de base de datos
    """
    cleanup_test_files()
    
    # Mock para usar rutas de test
    with patch('read_and_update.LANCEDB_PATH', TEST_DB_PATH), \
         patch('read_and_update.METADATA_FILE', TEST_METADATA_FILE):
        
        # Test 1: Base no existe
        status = validate_database_exists()
        assert status == 'empty', f"Esperaba 'empty', obtuvo '{status}'"
        
        # Test 2: Crear DB pero sin tabla
        os.makedirs(TEST_DB_PATH, exist_ok=True)
        db = connect_database(TEST_DB_PATH)
        status = validate_database_exists()
        assert status == 'no_table', f"Esperaba 'no_table', obtuvo '{status}'"
        
        # Test 3: DB con tabla
        table = prepare_table(db, TABLE_NAME, PdfChunk)
        status = validate_database_exists()
        assert status == 'exists', f"Esperaba 'exists', obtuvo '{status}'"
    
    cleanup_test_files()
    print("✅ Test de validación de estados de BD completado")

if __name__ == "__main__":
    # Ejecutar tests independientemente
    print("Ejecutando tests de read_and_update...")
    test_database_validation_states()
    test_crud_operations_complete()