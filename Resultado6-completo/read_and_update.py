#!/usr/bin/env python3

import os
import re
import json
import sys
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import polars as pl
import lancedb
from lancedb.pydantic import LanceModel, Vector

# Reutilizar imports y constantes de load_and_check
from load_and_check import (
    LANCEDB_PATH, TABLE_NAME, EMBEDDING_DIM, METADATA_FILE, DOCUMENTOS_CSV, PDF_DIR,
    PdfChunk, connect_database, init_converter_and_embedder,
    load_documentos_info, save_metadata_json, load_metadata_json,
    process_pdfs, insert_records, create_vector_index, create_fts_index,
    ensure_pdf_dir_exists, list_pdf_files, select_pdfs, prepare_table
)


# ============================================================================
# ZONA 1: CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# ============================================================================
# ZONA 2: FUNCIONES DE UTILIDAD Y VALIDACIÓN
# ============================================================================

def print_header(title: str) -> None:
    """Imprime un encabezado formateado"""
    separator = "=" * 60
    print(f"\n{separator}")
    print(f"  {title}")
    print(f"{separator}\n")


def print_success(message: str) -> None:
    """Imprime mensaje de éxito"""
    print(f"✓ {message}")


def print_error(message: str) -> None:
    """Imprime mensaje de error"""
    print(f"✗ {message}")


def print_warning(message: str) -> None:
    """Imprime mensaje de advertencia"""
    print(f"⚠ {message}")


def print_info(message: str) -> None:
    """Imprime mensaje informativo"""
    print(f"ℹ {message}")


def validate_database_exists() -> str:
    """Valida que la base de datos vectorial existe
    
    Returns:
        str: 'exists' si existe, 'empty' si no existe, 'no_table' si existe DB pero sin tabla
    """
    db_path = Path(LANCEDB_PATH)
    metadata_path = Path(METADATA_FILE)
    
    if not db_path.exists():
        print_warning("No se encontró base de datos vectorial")
        print_info(f"Ubicación esperada: {LANCEDB_PATH}")
        return 'empty'
    
    # Verificar si la tabla existe
    try:
        db = connect_database(LANCEDB_PATH)
        if TABLE_NAME not in db.table_names():
            print_warning("Base de datos encontrada pero sin tabla de documentos")
            print_info(f"Tabla esperada: {TABLE_NAME}")
            return 'no_table'
    except Exception as e:
        print_error(f"Error accediendo a la base de datos: {e}")
        return 'empty'
    
    if not metadata_path.exists():
        print_warning(f"Archivo de metadatos no encontrado: {METADATA_FILE}")
        print_info("Se utilizarán datos directos de la base vectorial")
    
    return 'exists'


def create_vector_database() -> bool:
    """Crea una nueva base de datos vectorial usando la lógica de load_and_check"""
    try:
        print_header("CREAR NUEVA BASE VECTORIAL")
        
        # Validar directorio de PDFs
        try:
            pdf_dir = ensure_pdf_dir_exists(PDF_DIR)
        except FileNotFoundError as e:
            print_error(str(e))
            return False
        
        # Cargar información de documentos
        docs_info = load_documentos_info()
        if not docs_info:
            print_error("No se pudo cargar información de documentos desde CSV")
            print_info("Verifica que 'Documentos.csv' existe y tiene el formato correcto")
            return False
        
        # Listar archivos disponibles
        try:
            pdf_files = list_pdf_files(pdf_dir, docs_info)
        except FileNotFoundError as e:
            print_error(str(e))
            return False
        
        if not pdf_files:
            print_error("No hay archivos PDF válidos para procesar")
            return False
        
        print_info(f"Encontrados {len(pdf_files)} documentos válidos")
        
        # Seleccionar documentos a procesar
        selected_pdfs = select_pdfs(pdf_files, docs_info)
        
        print_info(f"Seleccionados {len(selected_pdfs)} documentos para procesamiento")
        
        # Crear base de datos y tabla
        db = connect_database(LANCEDB_PATH)
        table = prepare_table(db, TABLE_NAME, PdfChunk)
        
        # Inicializar converter y embedder
        converter, embedder = init_converter_and_embedder("nomic-embed-text-v2", EMBEDDING_DIM)
        
        # Procesar PDFs
        print_info("Iniciando procesamiento de documentos...")
        records, doc_word_counts, metadata = process_pdfs(
            selected_pdfs, pdf_dir, converter, embedder, EMBEDDING_DIM, docs_info
        )
        
        if not records:
            print_error("No se pudieron generar registros de los documentos")
            return False
        
        # Insertar registros
        total = insert_records(table, records)
        
        # Crear índices
        print_info("Creando índices vectoriales...")
        create_vector_index(table, total, EMBEDDING_DIM)
        create_fts_index(table)
        
        # Guardar metadatos
        save_metadata_json(metadata)
        
        print_success(f"Base vectorial creada exitosamente con {total} registros")
        print_success(f"Metadatos guardados en '{METADATA_FILE}'")
        
        return True
        
    except Exception as e:
        print_error(f"Error creando base vectorial: {e}")
        return False


def handle_empty_database() -> bool:
    """Maneja el caso de base de datos inexistente o vacía"""
    print_header("BASE VECTORIAL NO ENCONTRADA")
    print_info("No existe una base de datos vectorial actualmente")
    print_info("Opciones disponibles:")
    print("  1. Crear nueva base vectorial")
    print("  2. Salir del programa")
    
    choice = get_valid_input("Selecciona una opción (1-2): ", ["1", "2"])
    
    if choice == "1":
        return create_vector_database()
    else:
        print_info("Saliendo del programa")
        return False


def get_valid_input(prompt: str, valid_options: List[str] = None, allow_empty: bool = False) -> str:
    """Obtiene input del usuario con validación"""
    while True:
        try:
            user_input = input(prompt).strip()
            
            if not user_input and allow_empty:
                return user_input
            
            if not user_input:
                print_warning("Entrada vacía no permitida. Inténtalo de nuevo.")
                continue
            
            if valid_options and user_input.lower() not in [opt.lower() for opt in valid_options]:
                print_warning(f"Opción inválida. Opciones válidas: {', '.join(valid_options)}")
                continue
                
            return user_input.strip()
            
        except KeyboardInterrupt:
            print("\nOperación cancelada por el usuario")
            sys.exit(0)


def format_doc_preview(text: str, max_chars: int = 100) -> str:
    """Formatea preview de texto para mostrar"""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


# ============================================================================
# ZONA 3: FUNCIONES DE CONEXIÓN Y CARGA DE DATOS
# ============================================================================

def load_vector_database() -> Tuple[lancedb.db.LanceDBConnection, lancedb.table.LanceTable]:
    """Carga y conecta a la base de datos vectorial"""
    try:
        db = connect_database(LANCEDB_PATH)
        
        if TABLE_NAME not in db.table_names():
            print_error(f"Tabla '{TABLE_NAME}' no encontrada en la base de datos")
            print_info("Ejecuta 'python load_and_check.py' para crear la tabla")
            sys.exit(1)
        
        table = db.open_table(TABLE_NAME)
        print_success(f"Base de datos vectorial cargada: {len(table)} registros")
        return db, table
        
    except Exception as e:
        print_error(f"Error cargando base de datos: {e}")
        sys.exit(1)


def load_documents_data() -> Tuple[Dict, Dict]:
    """Carga metadatos JSON y información de documentos CSV"""
    metadata = load_metadata_json()
    docs_info = load_documentos_info()
    
    if metadata:
        print_success(f"Metadatos cargados: {len(metadata.get('documents', {}))} documentos")
    else:
        print_warning("Metadatos JSON no disponibles")
    
    if docs_info:
        print_success(f"Información CSV cargada: {len(docs_info)} documentos")
    else:
        print_warning("Información de documentos CSV no disponible")
    
    return metadata, docs_info


def get_table_dataframe(table: lancedb.table.LanceTable) -> pl.DataFrame:
    """Convierte tabla LanceDB a DataFrame Polars"""
    try:
        df = pl.from_arrow(table.to_arrow())
        print_success(f"DataFrame cargado: {df.height} filas, {df.width} columnas")
        return df
    except Exception as e:
        print_error(f"Error convirtiendo tabla a DataFrame: {e}")
        sys.exit(1)


# ============================================================================
# ZONA 4: FUNCIONES DE LISTADO Y VISUALIZACIÓN
# ============================================================================

def list_vectorized_documents(metadata: Dict, df: pl.DataFrame) -> List[str]:
    """Lista documentos vectorizados basándose en metadatos o DataFrame"""
    if metadata and "processing_order" in metadata:
        # Usar orden de procesamiento del JSON
        docs = metadata["processing_order"]
        print_info(f"Usando orden de procesamiento desde metadatos: {len(docs)} documentos")
        source = "metadatos"
    else:
        # Fallback al orden natural de la base de datos
        docs = df.select("pdf_name").unique().to_series().to_list()
        print_warning("Sin metadatos disponibles, usando orden de base de datos")
        source = "base de datos"
    
    print(f"\nDOCUMENTOS VECTORIZADOS (desde {source}):")
    print("-" * 60)
    
    for i, doc_name in enumerate(docs, 1):
        # Obtener información del documento
        if metadata and doc_name in metadata.get("documents", {}):
            doc_meta = metadata["documents"][doc_name]
            titulo = doc_meta.get('titulo', 'Sin título')
            fuente = doc_meta.get('fuente', 'Sin fuente')
            chunks = doc_meta.get('total_chunks', 0)
            palabras = doc_meta.get('total_words', 0)
            
            print(f"{i:2d}. {doc_name}")
            print(f"     Título: {titulo}")
            print(f"     Fuente: {fuente}")
            print(f"     Chunks: {chunks} | Palabras: {palabras}")
        else:
            # Información básica desde DataFrame
            doc_chunks = df.filter(pl.col("pdf_name") == doc_name).height
            print(f"{i:2d}. {doc_name}")
            print(f"     Chunks: {doc_chunks}")
        print()
    
    return docs


def list_available_documents_for_creation(docs_info: Dict, vectorized_docs: List[str]) -> List[str]:
    """Lista documentos disponibles en CSV que no están vectorizados"""
    if not docs_info:
        print_error("No hay información de documentos disponible en CSV")
        return []
    
    available_docs = []
    print(f"\nDOCUMENTOS DISPONIBLES PARA AGREGAR:")
    print("-" * 60)
    
    for i, (filename, info) in enumerate(docs_info.items(), 1):
        if filename not in vectorized_docs:
            # Verificar si el archivo PDF existe
            pdf_path = os.path.join("docs", filename)
            if os.path.exists(pdf_path):
                available_docs.append(filename)
                print(f"{len(available_docs):2d}. {filename}")
                print(f"     Título: {info.get('titulo', 'Sin título')}")
                print(f"     Fuente: {info.get('fuente', 'Sin fuente')}")
                print(f"     ID: {info.get('id', 'Sin ID')}")
                print()
            else:
                print(f"{i:2d}. {filename} (ARCHIVO NO ENCONTRADO)")
                print()
    
    if not available_docs:
        print_warning("No hay documentos nuevos disponibles para agregar")
    else:
        print_success(f"Total documentos disponibles para agregar: {len(available_docs)}")
    
    return available_docs


# ============================================================================
# ZONA 5: OPERACIONES CRUD - CREATE
# ============================================================================

def create_new_document(available_docs: List[str], docs_info: Dict, db: lancedb.db.LanceDBConnection, 
                       table: lancedb.table.LanceTable, metadata: Dict) -> bool:
    """Agrega nuevo documento a la base vectorial"""
    if not available_docs:
        print_error("No hay documentos disponibles para agregar")
        return False
    
    print_header("AGREGAR NUEVO DOCUMENTO")
    
    # Mostrar documentos disponibles
    for i, doc in enumerate(available_docs, 1):
        info = docs_info.get(doc, {})
        print(f"{i:2d}. {doc}")
        print(f"    Título: {info.get('titulo', 'Sin título')}")
    
    # Seleccionar documento
    choice = get_valid_input("\nSelecciona el número del documento a agregar (o 'c' para cancelar): ")
    
    if choice.lower() == 'c':
        print_warning("Operación cancelada")
        return False
    
    try:
        doc_index = int(choice) - 1
        if doc_index < 0 or doc_index >= len(available_docs):
            print_error("Número de documento inválido")
            return False
        
        selected_doc = available_docs[doc_index]
        print_info(f"Procesando documento: {selected_doc}")
        
        # Inicializar converter y embedder
        converter, embedder = init_converter_and_embedder("nomic-embed-text-v2", EMBEDDING_DIM)
        
        # Procesar el documento
        pdf_dir = "docs"
        records, doc_word_counts, new_metadata = process_pdfs([selected_doc], pdf_dir, converter, 
                                                             embedder, EMBEDDING_DIM, docs_info)
        
        if not records:
            print_error("No se pudieron generar registros para el documento")
            return False
        
        # Insertar registros
        print_info(f"Insertando {len(records)} registros...")
        insert_records(table, records)
        
        # Actualizar metadatos existentes
        if metadata:
            # Agregar al orden de procesamiento
            if "processing_order" not in metadata:
                metadata["processing_order"] = []
            metadata["processing_order"].append(selected_doc)
            
            # Agregar metadatos del documento
            if "documents" not in metadata:
                metadata["documents"] = {}
            
            doc_meta = new_metadata["documents"][selected_doc]
            doc_meta["order"] = len(metadata["processing_order"]) - 1
            metadata["documents"][selected_doc] = doc_meta
            
            # Actualizar timestamp
            metadata["processing_timestamp"] = datetime.now().isoformat()
            
            # Guardar metadatos actualizados
            save_metadata_json(metadata)
            print_success("Metadatos actualizados correctamente")
        
        # Preguntar sobre reindexación
        total_records = len(table)
        print_success(f"Documento '{selected_doc}' agregado exitosamente")
        print_info(f"Total registros en base: {total_records}")
        
        reindex = get_valid_input("¿Reindexar base vectorial para optimizar rendimiento? (y/N): ", allow_empty=True)
        
        if reindex.lower() == 'y':
            print_info("Recreando índices vectoriales...")
            create_vector_index(table, total_records, EMBEDDING_DIM)
            create_fts_index(table)
            print_success("Índices recreados correctamente")
        
        return True
        
    except ValueError:
        print_error("Entrada inválida. Debe ser un número.")
        return False
    except Exception as e:
        print_error(f"Error agregando documento: {e}")
        return False


# ============================================================================
# ZONA 6: OPERACIONES CRUD - READ
# ============================================================================

def read_document_details(vectorized_docs: List[str], metadata: Dict, df: pl.DataFrame) -> None:
    """Muestra detalles completos de un documento vectorizado"""
    if not vectorized_docs:
        print_error("No hay documentos vectorizados disponibles")
        return
    
    print_header("LEER DETALLES DE DOCUMENTO")
    
    # Mostrar lista numerada
    for i, doc in enumerate(vectorized_docs, 1):
        if metadata and doc in metadata.get("documents", {}):
            titulo = metadata["documents"][doc].get("titulo", "Sin título")
            print(f"{i:2d}. {doc} - {titulo[:50]}...")
        else:
            print(f"{i:2d}. {doc}")
    
    # Seleccionar documento
    choice = get_valid_input("\nSelecciona el número del documento a leer (o 'c' para cancelar): ")
    
    if choice.lower() == 'c':
        print_warning("Operación cancelada")
        return
    
    try:
        doc_index = int(choice) - 1
        if doc_index < 0 or doc_index >= len(vectorized_docs):
            print_error("Número de documento inválido")
            return
        
        selected_doc = vectorized_docs[doc_index]
        
        # Mostrar información detallada
        print_header(f"DETALLES: {selected_doc}")
        
        # Información de metadatos
        if metadata and selected_doc in metadata.get("documents", {}):
            doc_meta = metadata["documents"][selected_doc]
            print("METADATOS:")
            print(f"  Título: {doc_meta.get('titulo', 'Sin título')}")
            print(f"  Fuente: {doc_meta.get('fuente', 'Sin fuente')}")
            print(f"  Orden de procesamiento: {doc_meta.get('order', 'N/A')}")
            print(f"  Total palabras: {doc_meta.get('total_words', 0):,}")
            print(f"  Total chunks: {doc_meta.get('total_chunks', 0):,}")
            print(f"  Archivo: {doc_meta.get('file_path', 'N/A')}")
            print()
        
        # Información de la base de datos
        doc_data = df.filter(pl.col("pdf_name") == selected_doc)
        print("INFORMACIÓN DE BASE VECTORIAL:")
        print(f"  Registros almacenados: {doc_data.height:,}")
        
        if doc_data.height > 0:
            # Calcular estadísticas
            word_counts = (
                doc_data
                .with_columns(pl.col("text").str.split(" ").list.len().alias("word_count"))
                .select("word_count")
                .to_series()
            )
            
            total_words_db = word_counts.sum()
            avg_words_chunk = word_counts.mean()
            min_words = word_counts.min()
            max_words = word_counts.max()
            
            print(f"  Palabras totales (calculadas): {total_words_db:,}")
            print(f"  Promedio palabras por chunk: {avg_words_chunk:.1f}")
            print(f"  Rango palabras por chunk: {min_words} - {max_words}")
            print()
        
        # Mostrar preview de chunks
        show_content = get_valid_input("¿Mostrar contenido de chunks? (y/N): ", allow_empty=True)
        
        if show_content.lower() == 'y':
            print("CONTENIDO DE CHUNKS:")
            print("-" * 60)
            
            chunks_data = (
                doc_data
                .sort("chunk_index")
                .select(["chunk_index", "text"])
            )
            
            for row in chunks_data.iter_rows(named=True):
                chunk_idx = row["chunk_index"]
                text = row["text"]
                preview = format_doc_preview(text, 200)
                
                print(f"\nChunk {chunk_idx}:")
                print(f"  Palabras: {len(text.split())}")
                print(f"  Caracteres: {len(text)}")
                print(f"  Preview: {preview}")
        
        # Mostrar texto completo concatenado
        show_full = get_valid_input("\n¿Mostrar texto completo concatenado? (y/N): ", allow_empty=True)
        
        if show_full.lower() == 'y':
            full_texts = (
                doc_data
                .sort("chunk_index")
                .select("text")
                .to_series()
                .to_list()
            )
            
            print("\nTEXTO COMPLETO:")
            print("=" * 80)
            print("\n\n".join(full_texts))
            print("=" * 80)
        
    except ValueError:
        print_error("Entrada inválida. Debe ser un número.")
    except Exception as e:
        print_error(f"Error leyendo documento: {e}")


# ============================================================================
# ZONA 7: OPERACIONES CRUD - UPDATE
# ============================================================================

def update_document_metadata(vectorized_docs: List[str], metadata: Dict) -> bool:
    """Actualiza metadatos de un documento vectorizado"""
    if not vectorized_docs:
        print_error("No hay documentos vectorizados disponibles")
        return False
    
    if not metadata:
        print_error("No hay archivo de metadatos disponible para actualizar")
        print_info("Los metadatos se crean automáticamente al procesar documentos")
        return False
    
    print_header("ACTUALIZAR METADATOS DE DOCUMENTO")
    
    # Mostrar lista numerada
    for i, doc in enumerate(vectorized_docs, 1):
        if doc in metadata.get("documents", {}):
            titulo = metadata["documents"][doc].get("titulo", "Sin título")
            print(f"{i:2d}. {doc} - {titulo[:50]}...")
        else:
            print(f"{i:2d}. {doc} (sin metadatos)")
    
    # Seleccionar documento
    choice = get_valid_input("\nSelecciona el número del documento a actualizar (o 'c' para cancelar): ")
    
    if choice.lower() == 'c':
        print_warning("Operación cancelada")
        return False
    
    try:
        doc_index = int(choice) - 1
        if doc_index < 0 or doc_index >= len(vectorized_docs):
            print_error("Número de documento inválido")
            return False
        
        selected_doc = vectorized_docs[doc_index]
        
        # Obtener metadatos actuales
        if selected_doc not in metadata.get("documents", {}):
            print_error(f"No hay metadatos para el documento: {selected_doc}")
            return False
        
        doc_meta = metadata["documents"][selected_doc]
        
        print("\nMETADATOS ACTUALES:")
        print(f"  Título: {doc_meta.get('titulo', 'Sin título')}")
        print(f"  Fuente: {doc_meta.get('fuente', 'Sin fuente')}")
        
        print("\nCAMPOS ACTUALIZABLES:")
        print("1. Título")
        print("2. Fuente")
        print("3. Ambos")
        
        field_choice = get_valid_input("Selecciona qué actualizar (1-3): ", ["1", "2", "3"])
        
        changes_made = False
        
        if field_choice in ["1", "3"]:
            # Actualizar título
            current_title = doc_meta.get('titulo', '')
            print(f"\nTítulo actual: {current_title}")
            new_title = get_valid_input("Nuevo título (Enter para mantener actual): ", allow_empty=True)
            
            if new_title:
                doc_meta['titulo'] = new_title
                changes_made = True
                print_success(f"Título actualizado: {new_title}")
        
        if field_choice in ["2", "3"]:
            # Actualizar fuente
            current_source = doc_meta.get('fuente', '')
            print(f"\nFuente actual: {current_source}")
            new_source = get_valid_input("Nueva fuente (Enter para mantener actual): ", allow_empty=True)
            
            if new_source:
                doc_meta['fuente'] = new_source
                changes_made = True
                print_success(f"Fuente actualizada: {new_source}")
        
        if changes_made:
            # Actualizar timestamp de modificación
            metadata["processing_timestamp"] = datetime.now().isoformat()
            
            # Guardar metadatos actualizados
            save_metadata_json(metadata)
            print_success(f"Metadatos del documento '{selected_doc}' actualizados correctamente")
            return True
        else:
            print_info("No se realizaron cambios")
            return False
        
    except ValueError:
        print_error("Entrada inválida. Debe ser un número.")
        return False
    except Exception as e:
        print_error(f"Error actualizando metadatos: {e}")
        return False


# ============================================================================
# ZONA 8: OPERACIONES CRUD - DELETE
# ============================================================================

def delete_document_from_database(vectorized_docs: List[str], metadata: Dict, 
                                 table: lancedb.table.LanceTable) -> bool:
    """Elimina un documento de la base vectorial"""
    if not vectorized_docs:
        print_error("No hay documentos vectorizados disponibles")
        return False
    
    print_header("ELIMINAR DOCUMENTO DE BASE VECTORIAL")
    print_warning("¡ATENCIÓN! Esta operación eliminará permanentemente el documento de la base vectorial.")
    print_info("Los archivos PDF originales no serán afectados.")
    
    # Mostrar lista numerada
    for i, doc in enumerate(vectorized_docs, 1):
        if metadata and doc in metadata.get("documents", {}):
            titulo = metadata["documents"][doc].get("titulo", "Sin título")
            chunks = metadata["documents"][doc].get("total_chunks", 0)
            print(f"{i:2d}. {doc} - {titulo[:40]}... ({chunks} chunks)")
        else:
            # Contar chunks desde la tabla
            df = pl.from_arrow(table.to_arrow())
            chunks = df.filter(pl.col("pdf_name") == doc).height
            print(f"{i:2d}. {doc} ({chunks} chunks)")
    
    # Seleccionar documento
    choice = get_valid_input("\nSelecciona el número del documento a ELIMINAR (o 'c' para cancelar): ")
    
    if choice.lower() == 'c':
        print_warning("Operación cancelada")
        return False
    
    try:
        doc_index = int(choice) - 1
        if doc_index < 0 or doc_index >= len(vectorized_docs):
            print_error("Número de documento inválido")
            return False
        
        selected_doc = vectorized_docs[doc_index]
        
        # Mostrar información detallada antes de confirmar
        print(f"\nDOCUMENTO A ELIMINAR: {selected_doc}")
        
        if metadata and selected_doc in metadata.get("documents", {}):
            doc_meta = metadata["documents"][selected_doc]
            print(f"  Título: {doc_meta.get('titulo', 'Sin título')}")
            print(f"  Chunks: {doc_meta.get('total_chunks', 0)}")
            print(f"  Palabras: {doc_meta.get('total_words', 0):,}")
        
        # Confirmación doble
        confirm1 = get_valid_input(f"\n¿Estás SEGURO de eliminar '{selected_doc}'? (escribe 'ELIMINAR' para confirmar): ")
        
        if confirm1 != 'ELIMINAR':
            print_warning("Eliminación cancelada - confirmación incorrecta")
            return False
        
        confirm2 = get_valid_input("Confirmación final - esta acción NO se puede deshacer (y/N): ", ["y", "Y", "n", "N"])
        
        if confirm2.lower() != 'y':
            print_warning("Eliminación cancelada")
            return False
        
        print_info(f"Eliminando documento '{selected_doc}' de la base vectorial...")
        
        # Eliminar registros de la tabla usando filtro SQL
        try:
            # LanceDB utiliza predicados SQL para eliminación
            rows_deleted = table.delete(f"pdf_name = '{selected_doc}'")
            print_success(f"Eliminados {rows_deleted} registros de la base vectorial")
        except Exception as e:
            print_error(f"Error eliminando registros: {e}")
            return False
        
        # Actualizar metadatos
        if metadata:
            # Remover de processing_order
            if "processing_order" in metadata and selected_doc in metadata["processing_order"]:
                metadata["processing_order"].remove(selected_doc)
            
            # Remover de documents
            if "documents" in metadata and selected_doc in metadata["documents"]:
                del metadata["documents"][selected_doc]
            
            # Reordenar índices en processing_order
            if "processing_order" in metadata:
                for i, doc_name in enumerate(metadata["processing_order"]):
                    if doc_name in metadata.get("documents", {}):
                        metadata["documents"][doc_name]["order"] = i
            
            # Actualizar timestamp
            metadata["processing_timestamp"] = datetime.now().isoformat()
            
            # Guardar metadatos actualizados
            save_metadata_json(metadata)
            print_success("Metadatos actualizados correctamente")
        
        # Preguntar sobre reindexación después de eliminar
        reindex = get_valid_input("¿Reindexar base vectorial para optimizar rendimiento? (y/N): ", allow_empty=True)
        
        if reindex.lower() == 'y':
            try:
                total_records = len(table)
                print_info("Recreando índices vectoriales...")
                create_vector_index(table, total_records, EMBEDDING_DIM)
                create_fts_index(table)
                print_success("Índices recreados correctamente")
            except Exception as e:
                print_warning(f"Error reindexando: {e}")
        
        print_success(f"Documento '{selected_doc}' eliminado exitosamente")
        print_info(f"Total registros restantes: {len(table)}")
        
        return True
        
    except ValueError:
        print_error("Entrada inválida. Debe ser un número.")
        return False
    except Exception as e:
        print_error(f"Error eliminando documento: {e}")
        return False


# ============================================================================
# ZONA 9: MENÚ PRINCIPAL E INTERFAZ
# ============================================================================

def show_main_menu() -> None:
    """Muestra el menú principal de opciones"""
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║               GESTIÓN BASE VECTORIAL                     ║")
    print(f"╚══════════════════════════════════════════════════════════╝\n")
    
    print("OPERACIONES CRUD:")
    print("  1. Leer detalles de documento (READ)")
    print("  2. Agregar nuevo documento (CREATE)")
    print("  3. Actualizar metadatos (UPDATE)")
    print("  4. Eliminar documento (DELETE)")
    print("  0. Salir")


def show_database_statistics(metadata: Dict, df: pl.DataFrame) -> None:
    """Muestra estadísticas generales de la base vectorial"""
    print_header("ESTADÍSTICAS DE BASE VECTORIAL")
    
    # Estadísticas básicas
    total_records = df.height
    total_documents = df.select("pdf_name").n_unique()
    
    print("GENERAL:")
    print(f"  Total registros (chunks): {total_records:,}")
    print(f"  Total documentos: {total_documents:,}")
    print(f"  Dimensiones vectoriales: {EMBEDDING_DIM}")
    print()
    
    # Estadísticas de palabras
    word_counts = (
        df
        .with_columns(pl.col("text").str.split(" ").list.len().alias("word_count"))
        .select("word_count")
        .to_series()
    )
    
    total_words = word_counts.sum()
    avg_words = word_counts.mean()
    
    print("CONTENIDO:")
    print(f"  Total palabras: {total_words:,}")
    if total_documents > 0:
        print(f"  Promedio palabras por chunk: {avg_words:.1f}")
        print(f"  Promedio palabras por documento: {total_words / total_documents:.1f}")
    else:
        print("  Base de datos vacía - sin documentos")
    print()
    
    # Distribución por documento
    if total_documents > 0:
        doc_stats = (
            df
            .group_by("pdf_name")
            .agg([
                pl.len().alias("chunks"),
                pl.col("text").str.split(" ").list.len().sum().alias("words")
            ])
            .sort("words", descending=True)
        )
        
        print("DISTRIBUCIÓN POR DOCUMENTO:")
        for row in doc_stats.head(10).iter_rows(named=True):
            doc_name = row["pdf_name"]
            chunks = row["chunks"]
            words = row["words"]
            
            # Obtener título si está disponible
            titulo = "Sin título"
            if metadata and doc_name in metadata.get("documents", {}):
                titulo = metadata["documents"][doc_name].get("titulo", "Sin título")
            
            print(f"  {doc_name}")
            print(f"    Título: {titulo[:60]}...")
            print(f"    Chunks: {chunks:,} | Palabras: {words:,}")
            print()
    else:
        print("DISTRIBUCIÓN POR DOCUMENTO:")
        print("  (Sin documentos en la base)")
        print()
    
    # Información de metadatos
    if metadata:
        processing_time = metadata.get("processing_timestamp", "N/A")
        embedding_model = metadata.get("embedding_model", "N/A")
        
        print("METADATOS:")
        print(f"  Último procesamiento: {processing_time}")
        print(f"  Modelo embedding: {embedding_model}")
        print(f"  Archivo metadatos: {METADATA_FILE}")
    else:
        print_warning("Sin archivo de metadatos disponible")


def reindex_vector_database(table: lancedb.table.LanceTable) -> bool:
    """Reindexiza la base vectorial para optimizar rendimiento"""
    print_header("REINDEXAR BASE VECTORIAL")
    
    print_warning("La reindexación puede tomar tiempo dependiendo del tamaño de la base")
    print_info("Se recomienda después de agregar/eliminar documentos")
    
    confirm = get_valid_input("¿Proceder con la reindexación? (y/N): ", allow_empty=True)
    
    if confirm.lower() != 'y':
        print_warning("Reindexación cancelada")
        return False
    
    try:
        total_records = len(table)
        print_info(f"Iniciando reindexación de {total_records:,} registros...")
        
        # Recrear índices vectoriales
        print_info("Recreando índice vectorial...")
        create_vector_index(table, total_records, EMBEDDING_DIM)
        
        # Recrear índice FTS
        print_info("Recreando índice de texto completo...")
        create_fts_index(table)
        
        print_success("Reindexación completada exitosamente")
        return True
        
    except Exception as e:
        print_error(f"Error durante reindexación: {e}")
        return False


def main_menu_loop() -> None:
    """Loop principal del menú interactivo"""
    # Cargar datos iniciales
    db, table = load_vector_database()
    metadata, docs_info = load_documents_data()
    df = get_table_dataframe(table)
    
    # Mostrar información de la base al inicio
    show_database_statistics(metadata, df)
    vectorized_docs = list_vectorized_documents(metadata, df)
    
    while True:
        try:
            show_main_menu()
            
            choice = get_valid_input("\nSelecciona una opción (0-4): ", 
                                   ["0", "1", "2", "3", "4"])
            
            if choice == "0":
                print_info("¡Hasta luego!")
                sys.exit(0)
            
            elif choice == "1":
                # Leer detalles de documento
                read_document_details(vectorized_docs, metadata, df)
                input("\nPresiona Enter para continuar...")
            
            elif choice == "2":
                # Agregar nuevo documento
                available_docs = list_available_documents_for_creation(docs_info, vectorized_docs)
                
                if create_new_document(available_docs, docs_info, db, table, metadata):
                    # Recargar datos después de agregar
                    metadata, _ = load_documents_data()
                    df = get_table_dataframe(table)
                    vectorized_docs = list_vectorized_documents(metadata, df)
                
                input("\nPresiona Enter para continuar...")
            
            elif choice == "3":
                # Actualizar metadatos
                if update_document_metadata(vectorized_docs, metadata):
                    # Recargar metadatos después de actualizar
                    metadata, _ = load_documents_data()
                
                input("\nPresiona Enter para continuar...")
            
            elif choice == "4":
                # Eliminar documento
                if delete_document_from_database(vectorized_docs, metadata, table):
                    # Recargar datos después de eliminar
                    metadata, _ = load_documents_data()
                    df = get_table_dataframe(table)
                    vectorized_docs = list_vectorized_documents(metadata, df)
                
                input("\nPresiona Enter para continuar...")
        
        except KeyboardInterrupt:
            print("\nOperación interrumpida por el usuario")
            sys.exit(0)
        except Exception as e:
            print_error(f"Error inesperado: {e}")
            input("\nPresiona Enter para continuar...")


# ============================================================================
# ZONA 10: FUNCIÓN PRINCIPAL Y PUNTO DE ENTRADA
# ============================================================================

def main() -> None:
    """Función principal del script"""
    try:
        print_header("GESTIÓN DE BASE VECTORIAL")
        
        # Validar existencia de base de datos
        db_status = validate_database_exists()
        
        if db_status == 'empty':
            # No existe base de datos - ofrecer crear una nueva
            if not handle_empty_database():
                sys.exit(0)  # Usuario eligió salir o falló la creación
        elif db_status == 'no_table':
            # Existe DB pero no tabla - recrear tabla
            print_warning("La base de datos existe pero no contiene la tabla de documentos")
            recreate = get_valid_input("¿Recrear la tabla con documentos? (y/N): ", allow_empty=True)
            if recreate.lower() == 'y':
                if not create_vector_database():
                    sys.exit(1)
            else:
                print_info("No se puede proceder sin una tabla válida")
                sys.exit(0)
        else:
            # Base de datos existe y es válida
            print_success("Base de datos vectorial encontrada y validada")
        
        # Iniciar loop del menú principal
        main_menu_loop()
        
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario")
        sys.exit(0)
    except Exception as e:
        print_error(f"Error fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()