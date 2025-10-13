from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.embedder.ollama import OllamaEmbedder
import lancedb
import numpy as np
import polars as pl
import random
from textwrap import dedent
from typing import Iterator, Optional, Dict, List
import os
import ctranslate2 as ct2
from transformers import AutoTokenizer
import subprocess
import sys


class NLLBTranslator:
    """
    Clase para manejar traducción bidireccional español-inglés usando NLLB-200 con CTranslate2.
    """
    
    def __init__(
        self,
        model_dir: str = "nllb200_600M_int8", #"nllb-200-3.3B-ct2-int8",
        device: str = "auto",
        compute_type: str = None,
        beam_size: int = 4,
        length_penalty: float = 1.0
    ):
        """
        Inicializa el traductor NLLB.
        
        Args:
            model_dir: Directorio del modelo CTranslate2
            device: Dispositivo (auto, cpu, cuda)
            compute_type: Tipo de cómputo CT2
            beam_size: Tamaño del beam para la traducción
            length_penalty: Penalidad por longitud
        """
        self.model_dir = model_dir
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        
        # Auto-detectar dispositivo
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except:
                self.device = "cpu"
        else:
            self.device = device
        
        # Configurar compute_type
        self.compute_type = compute_type or ("int8_float16" if self.device == "cuda" else "int8")
        
        # Inicializar tokenizer y traductor
        self._init_translator()
    
    def _init_translator(self):
        """Inicializa el tokenizer y traductor."""
        try:
            # Usar tokenizer HF para NLLB
            hf_ckpt = "facebook/nllb-200-distilled-600M"
            self.tokenizer = AutoTokenizer.from_pretrained(hf_ckpt)
            
            # Cargar traductor CT2
            self.translator = ct2.Translator(
                self.model_dir, 
                device=self.device, 
                compute_type=self.compute_type
            )
            
            print(f"✓ Traductor NLLB cargado: {self.device} ({self.compute_type})")
            
        except Exception as e:
            print(f"✗ Error cargando traductor NLLB: {e}")
            self.translator = None
            self.tokenizer = None
    
    def translate_es_to_en(self, text: str) -> str:
        """Traduce de español a inglés."""
        if not self.translator or not text.strip():
            return text
        
        try:
            # Configurar tokenizer para español
            self.tokenizer.src_lang = "spa_Latn"
            source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
            
            # Traducir
            results = self.translator.translate_batch(
                [source_tokens],
                target_prefix=[["eng_Latn"]],
                beam_size=self.beam_size,
                length_penalty=self.length_penalty,
            )
            
            # Decodificar resultado (sin el token de idioma inicial)
            target_tokens = results[0].hypotheses[0][1:]
            translation = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens))
            
            return translation.strip()
            
        except Exception as e:
            print(f"⚠️ Error traduciendo ES->EN: {e}")
            return text
    
    def translate_en_to_es(self, text: str) -> str:
        """Traduce de inglés a español."""
        if not self.translator or not text.strip():
            return text
        
        try:
            # Configurar tokenizer para inglés
            self.tokenizer.src_lang = "eng_Latn"
            source_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text))
            
            # Traducir
            results = self.translator.translate_batch(
                [source_tokens],
                target_prefix=[["spa_Latn"]],
                beam_size=self.beam_size,
                length_penalty=self.length_penalty,
            )
            
            # Decodificar resultado (sin el token de idioma inicial)
            target_tokens = results[0].hypotheses[0][1:]
            translation = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(target_tokens))
            
            return translation.strip()
            
        except Exception as e:
            print(f"⚠️ Error traduciendo EN->ES: {e}")
            return text


class RAGAgent(Agent):
    """
    Clase RAGAgent que hereda de Agent de Agno y encapsula toda la lógica RAG
    para el chatbot Qhali con búsqueda híbrida en base de conocimiento PDF
    y pipeline de traducción español-inglés.
    """
    
    def __init__(
        self,
        model_id: str = "llama3.2",
        embedder_id: str = "nomic-embed-text-v2",
        embedding_dim: int = 768,
        lancedb_path: str = "tmp/lancedb",
        table_name: str = "docs_qa",
        translator_model_dir: str = "nllb200_600M_int8", #"nllb-200-3.3B-ct2-int8",
        max_history: int = 10,
        top_k_semantic: int = 5,
        top_k_keyword: int = 5,
        alpha: float = 0.3,
        similarity_threshold: float = 0.7,
        show_tool_calls: bool = False,
        markdown: bool = False,
        auto_warm_up: bool = True,
        enable_translation: bool = True,
        test_mode: bool = False,
        test_top_k_semantic: int = 50,
        test_top_k_keyword: int = 50,
        ignore_similarity_threshold: bool = False,
        silent_translation: bool = False,
        skip_chunk_translation: bool = False,
        # Advanced RAG parameters
        mode: str = "hybrid",
        adv_num_queries: int = 5,
        adv_top_k_per_query: int = 5,
        adv_merge_strategy: str = "vote+score",  # ["vote", "avg", "max", "vote+score"]
        adv_rerank_strategy: str = "mmr",       # ["none", "mmr"]
        adv_max_chunks: int = 5,
        adv_alpha: float = None,                 # override por modo advanced (opcional)
        adv_similarity_threshold: float = None   # override por modo advanced (opcional)
    ):
        """
        Inicializa el RAGAgent con todos los parámetros necesarios.

        Args:
            model_id: ID del modelo de Ollama a usar
            embedder_id: ID del modelo de embeddings
            embedding_dim: Dimensión de los embeddings
            lancedb_path: Ruta a la base de datos LanceDB
            table_name: Nombre de la tabla en LanceDB (docs_qa para PDFs)
            translator_model_dir: Directorio del modelo NLLB para traducción
            max_history: Número máximo de turnos en el historial
            top_k_semantic: Número de resultados semánticos
            top_k_keyword: Número de resultados por palabras clave
            alpha: Peso de la búsqueda semántica (0-1)
            similarity_threshold: Umbral de similitud mínimo
            show_tool_calls: Mostrar llamadas a herramientas
            markdown: Usar formato markdown
            auto_warm_up: Ejecutar warm-up automáticamente
            enable_translation: Habilitar pipeline de traducción
            test_mode: Activar modo de prueba con parámetros especiales
            test_top_k_semantic: Top K semántico para modo test (mayor cobertura)
            test_top_k_keyword: Top K keyword para modo test (mayor cobertura)
            ignore_similarity_threshold: Ignorar umbral de similitud en tests
            silent_translation: Suprimir logs de traducción (útil para testing)
            skip_chunk_translation: Omitir traducción de chunks (útil para testing)
            mode: Modo RAG ("hybrid" o "advanced")
            adv_num_queries: Número de reformulaciones para modo advanced
            adv_top_k_per_query: Top K por subconsulta en modo advanced
            adv_merge_strategy: Estrategia de fusión ("vote", "avg", "max", "vote+score")
            adv_rerank_strategy: Estrategia de reordenamiento ("none", "mmr")
            adv_max_chunks: Número máximo de chunks en modo advanced
            adv_alpha: Override de alpha para modo advanced
            adv_similarity_threshold: Override de umbral para modo advanced
        """
        # Configurar modelo y embedder
        self.model_id = model_id
        self.embedder_id = embedder_id
        self.embedding_dim = embedding_dim
        self.embedder = OllamaEmbedder(id=embedder_id, dimensions=embedding_dim)
        
        # Configurar LanceDB
        self.lancedb_path = lancedb_path
        self.table_name = table_name
        self.db = lancedb.connect(lancedb_path)
        self.tabla = self.db.open_table(table_name)
        
        # Inicializar traductor NLLB
        self.enable_translation = enable_translation
        if self.enable_translation:
            try:
                self.translator = NLLBTranslator(model_dir=translator_model_dir)
            except Exception as e:
                print(f"⚠️ No se pudo cargar el traductor NLLB: {e}")
                print("⚠️ Continuando sin traducción automática")
                self.enable_translation = False
                self.translator = None
        else:
            self.translator = None
        
        # Parámetros de búsqueda
        self.top_k_semantic = top_k_semantic
        self.top_k_keyword = top_k_keyword
        self.alpha = alpha
        self.similarity_threshold = similarity_threshold

        # Parámetros para modo test
        self.test_mode = test_mode
        self.test_top_k_semantic = test_top_k_semantic
        self.test_top_k_keyword = test_top_k_keyword
        self.ignore_similarity_threshold = ignore_similarity_threshold
        self.silent_translation = silent_translation
        self.skip_chunk_translation = skip_chunk_translation
        
        # Parámetros para modo Advanced RAG
        self.mode = mode if mode in {"hybrid", "advanced"} else "hybrid"
        self.adv_num_queries = adv_num_queries
        self.adv_top_k_per_query = adv_top_k_per_query
        self.adv_merge_strategy = adv_merge_strategy
        self.adv_rerank_strategy = adv_rerank_strategy
        self.adv_max_chunks = adv_max_chunks
        self.adv_alpha = adv_alpha
        self.adv_similarity_threshold = adv_similarity_threshold
        
        # Historial de conversación
        self.max_history = max_history
        self.historial_conversacion = ""
        self.separador_historial = "\n\n"
        
        # Respuestas introspectivas
        self.introspective_responses = [
            "A ver... esto no lo tengo muy aprendido, pero puedo intentarlo.",
            "Hmmm... esto no lo tengo a la mano, pero déjame ver qué puedo armar.",
            "Vaya, esto es interesante. No lo tengo memorizado, pero voy a construir algo.",
            "Esto no está en mis registros... pero puedo razonar una buena respuesta.",
            "Dame un segundo... esto no lo he respondido antes, pero veamos qué podemos sacar.",
            "Nunca me han preguntado esto así exactamente, pero suena interesante. Vamos a intentarlo.",
            "No tengo una respuesta directa para esto... pero puedo pensar en algo útil.",
            "No encontré esto en mi base de conocimientos, así que lo pensaré un poco.",
            "A ver, necesito conectar algunos conceptos antes de responder...",
            "Esto es nuevo para mí... Déjame ver cómo lo planteo.",
            "Hmmm... no hay datos directos sobre esto, pero voy a razonar un poco.",
            "Esto me hace pensar... Vamos a armar algo interesante.",
            "No lo tengo claro de inmediato, pero puedo intentar conectar algunas ideas.",
            "Veamos.. Esto es un reto... No tengo la respuesta exacta, pero puedo improvisar algo.",
            "No encontré algo preciso, pero puedo darle una vuelta interesante.",
            "Dame un momento mientras lo pienso.",
            "Déjame ver... No tengo una respuesta rápida, pero creo que puedo construir una buena.",
            "Necesito conectar algunas ideas antes de responder... Un momento.",
            "Interesante pregunta... veamos si puedo pero puedo formular algo basado en lo que sé.",
        ]
        
        # Inicializar el agente padre con las instrucciones de Qhali
        super().__init__(
            model=Ollama(id=model_id),
            instructions=dedent("""
            Eres la robot Qhali, la promotora de la salud en la Pontificia Universidad Católica del Perú. 
            Respondes sobre dudas y preocupaciones de los alumnos en formato conversacional, de forma empática y concisa.
            Tus respuestas son breves y fomentan el diálogo.
            
            - No incluyes enlaces.
            - Respondes en formato oración, sin listas o respuestas largas.
            - Si la pregunta es muy amplia, pides más detalles antes de responder.
            - Si el usuario agradece, Qhali se despide.

            Ejemplo:
            Usuario pregunta: '¿Cuántos litros de agua debo tomar al día?'
            Qhali responde: 'Dependiendo de tu peso y hábitos, podrías requerir entre 1.5 y 2 litros. ¿Quieres calcularlo juntos?'.
            """),
            show_tool_calls=show_tool_calls,
            markdown=markdown
        )
        
        # Ejecutar warm-up automáticamente si está habilitado
        if auto_warm_up:
            self._warm_up()
    
    def _warm_up(self):
        """
        Ejecuta una llamada mínima al agente para cargar el modelo y recursos
        antes de atender la primera petición real.
        """
        try:
            query = "Hola, ¿quién eres?"
            respuesta = ""
            # Una consulta corta y rápida
            _ = self.run(query, stream=False)
            # Enrutamiento según modo
            if self.mode == "hybrid":
                contexto = self._custom_hybrid_search(query=query, similarity_threshold=0.7, alpha=0.8)
            else:
                contexto = self._advanced_search(query=query, similarity_threshold=0.7, alpha=0.8)
        except Exception as e:
            print(f"Warm-up fallido: {e}")
        else:
            print(60*"=")
            print("Qhali:")
            for linea in contexto:
                print(linea, end="", flush=True)
                respuesta += linea
            print(" ")
            self._actualizar_historial(query, respuesta)
    
    def _search_semantic(self, query_vector, top_k=5):
        """Realiza una búsqueda semántica (vectorial) en LanceDB sobre chunks de PDFs."""
        try:
            results = (
                self.tabla.search(query=query_vector)
                .metric("cosine")
                .limit(top_k)
                .to_polars()
            )
        except Exception as e:
            print(f"⚠️ Error en búsqueda semántica: {e}")
            return pl.DataFrame()

        # Asegurar que "_distance" existe
        if "_distance" in results.columns:
            # Conversión correcta: LanceDB con metric("cosine") entrega distancia ≈ 1 - cos_sim
            # Para mapear linealmente cos_sim ∈ [-1,1] → [0,1] preservando orden completo:
            # similitud = 1 - (_distance / 2) = (cos_sim + 1) / 2
            # NOTA: NO usar clip para evitar aplanar valores negativos, mantener rango completo
            results = results.with_columns([
                (1.0 - 0.5 * pl.col("_distance")).alias("similitud"),
                (1.0 - pl.col("_distance")).alias("cosine_raw")  # Para comparación entre α
            ])
        else:
            results = results.with_columns([
                pl.lit(float('inf')).alias("_distance"),
                pl.lit(0).alias("similitud")  # Si falta, asumir similitud 0
            ])

        return results

    def _sanitize_fts_query(self, query):
        """
        Sanitiza una query para que sea compatible con FTS (Full Text Search).
        Remueve caracteres especiales que pueden causar errores de sintaxis.
        """
        if not query:
            return ""

        # Remover o reemplazar caracteres problemáticos para FTS
        sanitized = query

        # Remover signos de interrogación y exclamación
        sanitized = sanitized.replace('?', '').replace('!', '')

        # Reemplazar apostrofes con espacios para evitar contracciones problemáticas
        sanitized = sanitized.replace("'", ' ').replace("'", ' ')

        # Remover comillas dobles
        sanitized = sanitized.replace('"', '').replace('"', '').replace('"', '')

        # Remover otros caracteres especiales que pueden causar problemas
        sanitized = sanitized.replace('(', '').replace(')', '')
        sanitized = sanitized.replace('[', '').replace(']', '')
        sanitized = sanitized.replace('{', '').replace('}', '')

        # Limpiar espacios múltiples
        import re
        sanitized = re.sub(r'\s+', ' ', sanitized).strip()

        return sanitized

    def _search_keyword(self, query, top_k=5):
        """Realiza una búsqueda por palabras clave (BM25) en LanceDB sobre chunks de PDFs."""
        try:
            # Sanitizar la query para FTS
            sanitized_query = self._sanitize_fts_query(query)

            if not sanitized_query:
                print("⚠️ Query vacía después de sanitización, retornando resultados vacíos")
                return pl.DataFrame()

            results = (
                self.tabla.search(query=sanitized_query, query_type="fts", fts_columns="text")
                .limit(top_k)
                .to_polars()
            )
        except Exception as e:
            print(f"⚠️ Error en búsqueda por palabras clave: {e}")
            # En caso de error, retornar DataFrame vacío pero bien estructurado
            # para mantener compatibilidad con el resto del pipeline
            empty_df = pl.DataFrame(schema={'pdf_name': pl.String, 'titulo': pl.String, 'fuente': pl.String, 'chunk_index': pl.Int64, 'text': pl.String, '_score': pl.Float64})
            return empty_df

        # Asegurar que "_score" existe
        if "_score" in results.columns:
            min_score = results["_score"].min()
            max_score = results["_score"].max()

            # Evitar división por cero
            if min_score == max_score:
                results = results.with_columns([
                    pl.lit(1.0).alias("score_BM25"),  # Si solo hay un resultado, darle máxima puntuación
                    pl.col("_score").alias("bm25_raw")  # Mantener valor raw para comparación
                ])
            else:
                results = results.with_columns([
                    ((pl.col("_score") - min_score) / (max_score - min_score + 1e-8)).alias("score_BM25"),
                    pl.col("_score").alias("bm25_raw")  # Mantener valor raw para comparación
                ])
        else:
            results = results.with_columns([
                pl.lit(0).alias("_score"),
                pl.lit(0).alias("score_BM25"),  # Si falta, asumir puntaje 0
                pl.lit(0.0).alias("bm25_raw")  # Para compatibilidad
            ])

        return results

    def _combine_results(self, results_semantic, results_keyword, alpha):
        """Fusiona, alinea y pondera correctamente los resultados de ambas búsquedas para chunks de PDFs."""
        
        if alpha < 0 or alpha > 1:
            raise ValueError("El parámetro alpha debe estar entre 0 y 1")

        if alpha == 1:  # Solo búsqueda semántica - usar similitud directamente (ya en [0,1])
            if not results_semantic.is_empty():
                results_semantic = results_semantic.with_columns(pl.col("similitud").alias("score_final"))
            return results_semantic

        if alpha == 0:  # Solo búsqueda por palabras clave - usar score_BM25 directamente (ya normalizado)
            if not results_keyword.is_empty():
                results_keyword = results_keyword.with_columns(pl.col("score_BM25").alias("score_final"))
            return results_keyword
        
        try:
            # Crear clave única para cada chunk usando pdf_name + chunk_index
            if not results_semantic.is_empty():
                results_semantic = results_semantic.with_columns(
                    (pl.col("pdf_name").cast(pl.String) + "_" + pl.col("chunk_index").cast(pl.String)).alias("chunk_key")
                )
            if not results_keyword.is_empty():
                results_keyword = results_keyword.with_columns(
                    (pl.col("pdf_name").cast(pl.String) + "_" + pl.col("chunk_index").cast(pl.String)).alias("chunk_key")
                )
            
            # Seleccionar columnas relevantes para el merge
            semantic_cols = ["chunk_key", "pdf_name", "titulo", "fuente", "chunk_index", "text", "similitud"]
            keyword_cols = ["chunk_key", "pdf_name", "titulo", "fuente", "chunk_index", "text", "score_BM25"]
            
            # Filtrar columnas que existen
            semantic_cols = [col for col in semantic_cols if col in results_semantic.columns] if not results_semantic.is_empty() else []
            keyword_cols = [col for col in keyword_cols if col in results_keyword.columns] if not results_keyword.is_empty() else []
            
            # Fusionamos ambas búsquedas alineando por chunk_key
            if not results_semantic.is_empty() and not results_keyword.is_empty():
                results_combined = results_semantic.select(semantic_cols).join(
                    results_keyword.select(keyword_cols),
                    on="chunk_key",
                    how="full",  # Unión externa para incluir todos los chunks
                    suffix="_kw"  # Evitar conflictos de columnas duplicadas
                )
                
                # Limpiar columnas duplicadas: usar datos principales, rellenar con _kw donde falten
                for col in ['pdf_name', 'titulo', 'fuente', 'chunk_index', 'text']:
                    col_kw = col + '_kw'
                    if col in results_combined.columns and col_kw in results_combined.columns:
                        results_combined = results_combined.with_columns(
                            pl.col(col).fill_null(pl.col(col_kw)).alias(col)
                        ).drop(col_kw)
            elif not results_semantic.is_empty():
                results_combined = results_semantic.select(semantic_cols).with_columns(
                    pl.lit(0).cast(pl.Float64).alias("score_BM25")
                )
            elif not results_keyword.is_empty():
                results_combined = results_keyword.select(keyword_cols).with_columns(
                    pl.lit(0).cast(pl.Float64).alias("similitud")
                )
            else:
                return pl.DataFrame()

            # Llenamos valores nulos con 0 en similitud y score_BM25
            results_combined = results_combined.with_columns([
                pl.col("similitud").fill_null(0),
                pl.col("score_BM25").fill_null(0)
            ])

            # Función para normalización min-max por consulta usando polars
            def _minmax_per_query_pl(col_name):
                col = pl.col(col_name).cast(pl.Float64).fill_null(0.0)
                col_min = col.min()
                col_max = col.max()
                rng = col_max - col_min
                return pl.when(rng > 0).then((col - col_min) / rng).otherwise(1.0)

            # Normalizar ambos canales antes de combinar
            results_combined = results_combined.with_columns([
                _minmax_per_query_pl("similitud").alias("sim_norm"),
                _minmax_per_query_pl("score_BM25").alias("bm25_norm")
            ])

            # Cálculo corregido de `score_final` usando scores normalizados
            results_combined = results_combined.with_columns(
                (alpha * pl.col("sim_norm") + (1 - alpha) * pl.col("bm25_norm")).alias("score_final")
            )
        except Exception as e:
            print(f"⚠️ Error combinando resultados: {e}")
            return None

        return results_combined

    def retrieve_only(self, query, top_k_semantic=None, top_k_keyword=None, alpha=None):
        """
        Método especializado para evaluación que devuelve solo la lista ordenada de chunks
        recuperados con metadatos completos, sin generar contexto ni respuesta.
        
        Args:
            query: Query de búsqueda
            top_k_semantic: Número de resultados semánticos (usa por defecto self.top_k_semantic)
            top_k_keyword: Número de resultados por palabras clave (usa por defecto self.top_k_keyword)
            alpha: Peso de búsqueda semántica (usa por defecto self.alpha)
            
        Returns:
            list: Lista ordenada de diccionarios con metadatos de chunks:
                - pdf_name: Nombre del documento
                - chunk_index: Índice del chunk en el documento
                - chunk_id: ID único del chunk (pdf_name#chunk_index)
                - text: Contenido del chunk
                - score_final: Score final de relevancia
                - titulo: Título del documento (si existe)
                - fuente: Fuente del documento (si existe)
        """
        # Enrutamiento según modo
        if self.mode != "hybrid":
            return self.retrieve_only_advanced(
                query,
                top_k_per_query=self.adv_top_k_per_query,
                alpha=self.adv_alpha if self.adv_alpha is not None else (alpha if alpha is not None else self.alpha)
            )
        
        # Usar valores por defecto si no se proporcionan
        top_k_semantic = top_k_semantic or self.top_k_semantic
        top_k_keyword = top_k_keyword or self.top_k_keyword
        alpha = self.alpha if alpha is None else alpha

        # PASO 1: Traducir query de español a inglés (si está habilitado)
        search_query = query
        if self.enable_translation and self.translator:
            try:
                search_query = self.translator.translate_es_to_en(query)
                if not self.silent_translation:
                    print(f"🔄 Query traducida para retrieve_only: {query} → {search_query}")
            except Exception as e:
                if not self.silent_translation:
                    print(f"⚠️ Error en traducción ES->EN, usando query original: {e}")
                search_query = query

        # PASO 2: Generar embedding y realizar búsquedas
        query_vector = np.array(self.embedder.get_embedding(search_query), dtype=np.float32) if alpha > 0 else None

        # Realizar búsquedas según alpha
        results_semantic = self._search_semantic(query_vector, top_k=top_k_semantic) if alpha > 0 else pl.DataFrame()
        results_keyword = self._search_keyword(search_query, top_k=top_k_keyword) if alpha < 1 else pl.DataFrame()

        # Si ambas búsquedas fallan, retornar lista vacía
        if results_semantic.is_empty() and results_keyword.is_empty():
            return []

        # PASO 3: Fusionar, deduplicar y ponderar resultados
        merged_results = self._combine_results(results_semantic, results_keyword, alpha)
        if merged_results is None or merged_results.is_empty():
            return []

        # PASO 4: Ordenar por score final y preparar salida
        merged_results = merged_results.sort("score_final", descending=True)
        
        # Crear chunk_id único y seleccionar columnas relevantes
        merged_results = merged_results.with_columns([
            (pl.col("pdf_name").cast(pl.String) + "#" + pl.col("chunk_index").cast(pl.String)).alias("chunk_id")
        ])
        
        # Seleccionar y ordenar columnas para salida consistente
        output_columns = [
            "pdf_name", "chunk_index", "chunk_id", "text", "score_final"
        ]
        
        # Agregar columnas opcionales si existen
        optional_columns = ["titulo", "fuente"]
        for col in optional_columns:
            if col in merged_results.columns:
                output_columns.append(col)
        
        # Filtrar solo las columnas que realmente existen
        available_columns = [col for col in output_columns if col in merged_results.columns]
        
        # Convertir a lista de diccionarios
        return merged_results.select(available_columns).to_dicts()

    def _custom_hybrid_search(
        self, 
        query, 
        top_k_semantic=None, 
        top_k_keyword=None, 
        alpha=None, 
        similarity_threshold=None
    ):
        """
        Realiza una búsqueda híbrida combinando búsqueda semántica (vectorial) y 
        búsqueda de palabras clave (BM25) sobre chunks de PDFs con pipeline de traducción.
        
        Pipeline:
        1. Traduce la query de español a inglés (si está habilitado)
        2. Busca en la base de conocimiento en inglés
        3. Traduce los resultados relevantes de inglés a español
        4. Retorna el contexto en español
        """
        # Usar valores por defecto si no se proporcionan
        top_k_semantic = top_k_semantic or self.top_k_semantic
        top_k_keyword = top_k_keyword or self.top_k_keyword
        alpha = alpha if alpha is not None else self.alpha
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        # PASO 1: Traducir query de español a inglés
        search_query = query
        if self.enable_translation and self.translator:
            try:
                search_query = self.translator.translate_es_to_en(query)
                if not self.silent_translation:
                    print(f"🔄 Query traducida: {query} → {search_query}")
            except Exception as e:
                print(f"⚠️ Error en traducción ES->EN, usando query original: {e}")
                search_query = query
        
        # PASO 2: Buscar en la base de conocimiento usando la query en inglés
        # Generar el embedding de la consulta traducida solo si alpha > 0
        query_vector = np.array(self.embedder.get_embedding(search_query), dtype=np.float32) if alpha > 0 else None

        # Obtener resultados de ambas búsquedas según alpha
        results_semantic = self._search_semantic(query_vector, top_k=top_k_semantic) if alpha > 0 else pl.DataFrame()
        results_keyword = self._search_keyword(search_query, top_k=top_k_keyword) if alpha < 1 else pl.DataFrame()

        # Si ambas búsquedas fallan, retornar None
        if results_semantic.is_empty() and results_keyword.is_empty():
            return None

        # Fusionar, deduplicar y ponderar resultados
        results = self._combine_results(results_semantic, results_keyword, alpha)
        if results is None:
            return None

        # Ordenar y filtrar por score final
        results_ordenados = results.sort("score_final", descending=True)
        results_filtrados = results_ordenados.filter(pl.col("score_final") >= similarity_threshold)

        # Evitar incluir chunks con `score_final = 0.0`
        results_filtrados = results_filtrados.filter(pl.col("score_final") > 0.0)

        # Limitar el contexto a un número razonable de chunks
        if not results_filtrados.is_empty():
            max_chunks = min(len(results_filtrados), 5)  # Límite máximo de 5 chunks
            results_filtrados = results_filtrados.head(max_chunks)

        if results_filtrados.is_empty():
            return None
        
        # PASO 3: Preparar contexto y traducir de inglés a español
        contexto_chunks = []
        for row in results_filtrados.to_dicts():
            chunk_text = row["text"]
            pdf_name = row.get("pdf_name", "documento")
            titulo = row.get("titulo", "")
            
            # Traducir el chunk de inglés a español (solo si no se debe omitir)
            if self.enable_translation and self.translator and not self.skip_chunk_translation:
                try:
                    chunk_text_es = self.translator.translate_en_to_es(chunk_text)
                    if not self.silent_translation:
                        print(f"🔄 Chunk traducido: {pdf_name} chunk {row.get('chunk_index', 0)}")
                except Exception as e:
                    print(f"⚠️ Error traduciendo chunk EN->ES: {e}")
                    chunk_text_es = chunk_text
            else:
                chunk_text_es = chunk_text
            
            # Formatear chunk con metadatos
            if titulo:
                chunk_formateado = f"**{titulo}** ({pdf_name}):\n{chunk_text_es}"
            else:
                chunk_formateado = f"**{pdf_name}**:\n{chunk_text_es}"
            
            contexto_chunks.append(chunk_formateado)

        # Verificar si tenemos una respuesta de muy alta calidad (>= 0.8)
        best_score = results_filtrados["score_final"][0]
        if best_score >= 0.8 and len(contexto_chunks) == 1:
            # Respuesta directa de alta calidad - solo el mejor chunk
            contexto = contexto_chunks[0]
        else:
            # Concatenar múltiples chunks como contexto
            contexto = "\n\n".join(contexto_chunks)

        return contexto

    def _custom_hybrid_search_with_score(
        self, 
        query, 
        top_k_semantic=None, 
        top_k_keyword=None, 
        alpha=None, 
        similarity_threshold=None
    ):
        """
        Versión mejorada de la búsqueda híbrida que también devuelve el score de confianza
        e incluye el pipeline de traducción.
        
        Returns:
            dict: {"contexto": str, "best_score": float} o None si no encuentra resultados
        """
        # Usar valores por defecto si no se proporcionan
        top_k_semantic = top_k_semantic or self.top_k_semantic
        top_k_keyword = top_k_keyword or self.top_k_keyword
        alpha = alpha if alpha is not None else self.alpha
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        # PASO 1: Traducir query de español a inglés
        search_query = query
        if self.enable_translation and self.translator:
            try:
                search_query = self.translator.translate_es_to_en(query)
                if not self.silent_translation:
                    print(f"🔄 Query traducida: {query} → {search_query}")
            except Exception as e:
                print(f"⚠️ Error en traducción ES->EN, usando query original: {e}")
                search_query = query
        
        # PASO 2: Buscar en la base de conocimiento usando la query en inglés
        # Generar el embedding de la consulta traducida solo si alpha > 0
        query_vector = np.array(self.embedder.get_embedding(search_query), dtype=np.float32) if alpha > 0 else None

        # Obtener resultados de ambas búsquedas según alpha
        results_semantic = self._search_semantic(query_vector, top_k=top_k_semantic) if alpha > 0 else pl.DataFrame()
        results_keyword = self._search_keyword(search_query, top_k=top_k_keyword) if alpha < 1 else pl.DataFrame()

        # Si ambas búsquedas fallan, retornar None
        if results_semantic.is_empty() and results_keyword.is_empty():
            return None

        # Fusionar, deduplicar y ponderar resultados
        results = self._combine_results(results_semantic, results_keyword, alpha)
        if results is None:
            return None

        # Ordenar y filtrar por score final
        results_ordenados = results.sort("score_final", descending=True)
        results_filtrados = results_ordenados.filter(pl.col("score_final") >= similarity_threshold)

        # Evitar incluir chunks con `score_final = 0.0`
        results_filtrados = results_filtrados.filter(pl.col("score_final") > 0.0)

        # Limitar el contexto a un número razonable de chunks
        if not results_filtrados.is_empty():
            max_chunks = min(len(results_filtrados), 5)  # Límite máximo de 5 chunks
            results_filtrados = results_filtrados.head(max_chunks)

        if results_filtrados.is_empty():
            return None
        
        # Obtener el mejor score
        best_score = results_filtrados["score_final"][0]
        
        # PASO 3: Preparar contexto y traducir de inglés a español
        contexto_chunks = []
        for row in results_filtrados.to_dicts():
            chunk_text = row["text"]
            pdf_name = row.get("pdf_name", "documento")
            titulo = row.get("titulo", "")
            
            # Traducir el chunk de inglés a español (solo si no se debe omitir)
            if self.enable_translation and self.translator and not self.skip_chunk_translation:
                try:
                    chunk_text_es = self.translator.translate_en_to_es(chunk_text)
                    if not self.silent_translation:
                        print(f"🔄 Chunk traducido: {pdf_name} chunk {row.get('chunk_index', 0)}")
                except Exception as e:
                    print(f"⚠️ Error traduciendo chunk EN->ES: {e}")
                    chunk_text_es = chunk_text
            else:
                chunk_text_es = chunk_text
            
            # Formatear chunk con metadatos
            if titulo:
                chunk_formateado = f"**{titulo}** ({pdf_name}):\n{chunk_text_es}"
            else:
                chunk_formateado = f"**{pdf_name}**:\n{chunk_text_es}"
            
            contexto_chunks.append(chunk_formateado)

        # Verificar si tenemos una respuesta de muy alta calidad (>= 0.8)
        if best_score >= 0.8 and len(contexto_chunks) == 1:
            # Respuesta directa de alta calidad - solo el mejor chunk
            contexto = contexto_chunks[0]
        else:
            # Concatenar múltiples chunks como contexto
            contexto = "\n\n".join(contexto_chunks)

        return {
            "contexto": contexto,
            "best_score": best_score
        }

    def test_hybrid_search_with_score(
        self,
        query,
        alpha=None,
        override_top_k_semantic=None,
        override_top_k_keyword=None,
        force_return_results=False
    ):
        """
        Función especializada para testing que permite usar parámetros de test
        con mayor cobertura y sin filtros de umbral.

        Args:
            query: Query de búsqueda
            alpha: Peso de búsqueda semántica (usa alpha del test_mode si no se especifica)
            override_top_k_semantic: Override manual para top_k_semantic
            override_top_k_keyword: Override manual para top_k_keyword
            force_return_results: Forzar retorno de resultados sin importar score

        Returns:
            dict: {"contexto": str, "best_score": float} siempre devuelve algo en modo test
        """
        # Validación de modo
        if self.mode != "hybrid":
            raise RuntimeError("test_hybrid_search_with_score solo está disponible en modo 'hybrid'.")
        
        # Determinar parámetros a usar
        if self.test_mode:
            top_k_semantic = override_top_k_semantic or self.test_top_k_semantic
            top_k_keyword = override_top_k_keyword or self.test_top_k_keyword
            similarity_threshold = 0.0 if self.ignore_similarity_threshold else self.similarity_threshold
        else:
            top_k_semantic = override_top_k_semantic or self.top_k_semantic
            top_k_keyword = override_top_k_keyword or self.top_k_keyword
            similarity_threshold = self.similarity_threshold

        alpha = alpha if alpha is not None else self.alpha

        # PASO 1: Traducir query de español a inglés
        search_query = query
        if self.enable_translation and self.translator:
            try:
                search_query = self.translator.translate_es_to_en(query)
                if not self.silent_translation:
                    print(f"🔄 Query traducida: {query} → {search_query}")
            except Exception as e:
                print(f"⚠️ Error en traducción ES->EN, usando query original: {e}")
                search_query = query

        # PASO 2: Buscar en la base de conocimiento usando la query en inglés
        query_vector = np.array(self.embedder.get_embedding(search_query), dtype=np.float32) if alpha > 0 else None

        # Obtener resultados de ambas búsquedas según alpha
        results_semantic = self._search_semantic(query_vector, top_k=top_k_semantic) if alpha > 0 else pl.DataFrame()
        results_keyword = self._search_keyword(search_query, top_k=top_k_keyword) if alpha < 1 else pl.DataFrame()

        # Combinar y fusionar resultados
        results_combined = self._combine_results(results_semantic, results_keyword, alpha)

        if results_combined is None or results_combined.is_empty():
            if force_return_results or self.test_mode:
                # En modo test, intentar devolver algo aunque sea de baja calidad
                return {
                    "contexto": "No se encontraron resultados relevantes para esta consulta.",
                    "best_score": 0.0
                }
            return None

        # Ordenar por score final
        results_combined = results_combined.sort("score_final", descending=True)

        # En modo test, no filtrar por threshold a menos que force_return_results sea False
        if not force_return_results and not self.test_mode:
            results_filtrados = results_combined.filter(pl.col("score_final") >= similarity_threshold)
        else:
            results_filtrados = results_combined

        # En modo test, incluir más chunks y permitir score 0.0
        if self.test_mode or force_return_results:
            max_chunks = min(len(results_filtrados), 10)  # Más chunks en modo test
        else:
            # Evitar incluir chunks con `score_final = 0.0`
            results_filtrados = results_filtrados.filter(pl.col("score_final") > 0.0)
            max_chunks = min(len(results_filtrados), 5)

        if max_chunks > 0:
            results_filtrados = results_filtrados.head(max_chunks)

        if results_filtrados.is_empty():
            if force_return_results or self.test_mode:
                return {
                    "contexto": "No se encontraron resultados relevantes para esta consulta.",
                    "best_score": 0.0
                }
            return None

        # Obtener el mejor score
        best_score = results_filtrados["score_final"][0]

        # PASO 3: Preparar contexto y traducir de inglés a español
        contexto_chunks = []
        for row in results_filtrados.to_dicts():
            chunk_text = row["text"]
            pdf_name = row.get("pdf_name", "documento")
            titulo = row.get("titulo", "")

            # Traducir el chunk de inglés a español (solo si no se debe omitir)
            if self.enable_translation and self.translator and not self.skip_chunk_translation:
                try:
                    chunk_text_es = self.translator.translate_en_to_es(chunk_text)
                    if not self.silent_translation:
                        print(f"🔄 Chunk traducido: {pdf_name} chunk {row.get('chunk_index', 0)}")
                except Exception as e:
                    print(f"⚠️ Error traduciendo chunk EN->ES: {e}")
                    chunk_text_es = chunk_text
            else:
                chunk_text_es = chunk_text

            # Formatear chunk con metadatos
            if titulo:
                chunk_formateado = f"**{titulo}** ({pdf_name}):\n{chunk_text_es}"
            else:
                chunk_formateado = f"**{pdf_name}**:\n{chunk_text_es}"

            contexto_chunks.append(chunk_formateado)

        # En modo test, siempre concatenar múltiples chunks
        contexto = "\n\n".join(contexto_chunks)

        return {
            "contexto": contexto,
            "best_score": best_score
        }

    def _introspective_thinking(self):
        """Devuelve una frase aleatoria de introspección antes de generar la respuesta junto con su índice."""
        # Seleccionar índice aleatorio (1-basado para coincidir con el manifest)
        index = random.randint(1, len(self.introspective_responses))
        # El texto está en la lista (0-basado), pero devolvemos el índice 1-basado
        text = self.introspective_responses[index - 1]
        return text, index

    def _actualizar_historial(self, usuario, qhali):
        """
        Agrega la nueva interacción al historial, asegurando que no exceda el límite de almacenamiento.
        """
        # Construcción eficiente de historial sin recrearlo cada vez
        nueva_interaccion = f"Usuario: {usuario}\nQhali: {qhali}"

        # Agregar al historial
        self.historial_conversacion += self.separador_historial + nueva_interaccion

        # Mantener un máximo de turnos en el historial
        historial_dividido = self.historial_conversacion.strip().split(self.separador_historial)
        if len(historial_dividido) > self.max_history:
            historial_dividido = historial_dividido[-self.max_history:]  # Mantener solo los últimos N turnos
        
        # Reconstrucción eficiente sin sobrecarga
        self.historial_conversacion = self.separador_historial.join(historial_dividido)

    def _construir_prompt(self, query, contexto, historial_externo=None, modo_evaluacion=False):
        """
        Construye el prompt asegurando una estructura clara entre historial, contexto y pregunta.
        
        Args:
            query: Pregunta del usuario
            contexto: Contexto recuperado del RAG
            historial_externo: Historial de conversación de la base de datos (opcional)
            modo_evaluacion: Si activar modo evaluación con salida estricta para mejorar EM
        """
        # Usar historial externo si se proporciona, sino usar el interno
        historial_a_usar = historial_externo if historial_externo else self.historial_conversacion
        
        # Prompt base
        prompt = f"""

        {"Conversación anterior: " + historial_a_usar if historial_a_usar else "Acabamos de iniciar la conversación"}

        Contexto Recuperado:
        {contexto if contexto else "Intentar responder"}

        Pregunta del usuario: {query}

        Basado en el contexto, responde de manera empática y concisa a la pregunta del usuario, básate en la conversación anterior cuando sea necesario.
        """
        
        # Añadir instrucciones de evaluación si está en modo evaluación
        if modo_evaluacion:
            prompt += """

        [INSTRUCCIÓN DE EVALUACIÓN]
        Devuelve SOLO la respuesta final en UNA LÍNEA, sin cortesías ni explicaciones adicionales.
        Si la respuesta es una frase corta presente en el contexto, repítela tal cual.
        Mantén la respuesta concisa y directa para evaluación exacta.
        """

        return prompt.strip()  # Eliminamos espacios en blanco adicionales

    def generate_answer_for_evaluation(self, query: str, contexto: str = None, use_json_format: bool = False) -> str:
        """
        Método especializado para generar respuestas durante evaluación con formato controlado
        para mejorar métricas como Exact Match.
        
        Args:
            query: Pregunta del usuario
            contexto: Contexto recuperado (opcional)
            use_json_format: Si usar formato JSON para mayor exactitud
            
        Returns:
            str: Respuesta generada optimizada para evaluación
        """
        if not contexto:
            # Buscar contexto si no se proporciona según modo
            if self.mode == "hybrid":
                search_result = self._custom_hybrid_search_with_score(
                    query=query,
                    similarity_threshold=0.0,  # Sin filtro para evaluación
                    alpha=self.alpha
                )
            else:
                search_result = self._advanced_search_with_score(
                    query=query,
                    similarity_threshold=0.0,
                    alpha=(self.adv_alpha if self.adv_alpha is not None else self.alpha)
                )
            contexto = search_result.get("contexto", "") if search_result else ""
        
        if use_json_format:
            # Modo JSON para máxima precisión
            prompt = f"""
            Contexto: {contexto if contexto else "Sin contexto específico"}
            
            Pregunta: {query}
            
            Responde SOLO en formato JSON estricto:
            {{"answer": "<respuesta exacta aquí>"}}
            
            La respuesta debe ser concisa, directa y basada en el contexto.
            """
            
            try:
                response = self.run(prompt.strip(), stream=False)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Intentar extraer JSON
                import json
                import re
                json_match = re.search(r'\{.*"answer".*:.*"(.*)".*\}', response_text, re.DOTALL)
                if json_match:
                    return json_match.group(1).strip()
                else:
                    # Fallback si no se puede parsear JSON
                    return response_text.strip()
                    
            except Exception as e:
                print(f"⚠️ Error en modo JSON: {e}")
                return ""
        else:
            # Modo evaluación estándar con salida controlada
            prompt = self._construir_prompt(query, contexto, historial_externo="", modo_evaluacion=True)
            
            try:
                response = self.run(prompt, stream=False)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Limpiar respuesta para evaluación
                response_text = response_text.strip()
                
                # Si la respuesta tiene múltiples líneas, tomar solo la primera línea significativa
                lines = [line.strip() for line in response_text.split('\n') if line.strip()]
                if lines:
                    return lines[0]
                else:
                    return response_text
                    
            except Exception as e:
                print(f"⚠️ Error generando respuesta de evaluación: {e}")
                return ""

    def responder_con_rag(self, query: str, historial_externo: str = None) -> str:
        """
        Genera una respuesta basada en el contexto recuperado de LanceDB o,
        en su defecto, invoca al agente tras una frase introspectiva.
        
        Args:
            query: Pregunta del usuario
            historial_externo: Historial de conversación de la base de datos (opcional)
            
        Returns:
            str: Respuesta generada
        """
        # Buscar contexto relevante en LanceDB según modo
        if self.mode == "hybrid":
            contexto = self._custom_hybrid_search(
                query=query, 
                similarity_threshold=self.similarity_threshold, 
                alpha=self.alpha
            )
        else:
            contexto = self._advanced_search(
                query=query,
                similarity_threshold=(self.adv_similarity_threshold or self.similarity_threshold),
                alpha=(self.adv_alpha if self.adv_alpha is not None else self.alpha)
            )
        
        # Inicializar respuesta
        respuesta = ""  

        if not contexto:    
            print("Qhali:")
            intro, intro_index = self._introspective_thinking()
            for i in intro:
                print(i, end="", flush=True)
            # Construcción del prompt con historial y contexto
            print(" ")
            prompt = self._construir_prompt(query, contexto, historial_externo)
            # Ejecutar la consulta con RAG
            response: Iterator = self.run(prompt, stream=True)
            if isinstance(response, Iterator):  # Si es un iterador, iteramos sobre él
                for resp in response:
                    respuesta += resp.content if resp.content else ""
                    print(resp.content, flush=True, end='')
                print(" ")
            else:
                respuesta = response.content if response.content else "No se generó respuesta."
        else:
            print("Qhali:")
            for linea in contexto.split("\n"):
                print(linea, end="", flush=True)
                respuesta += linea
            print(" ")
        
        self._actualizar_historial(query, respuesta)
        return respuesta

    def responder_con_rag_streaming(self, query: str, historial_externo: str = None) -> Iterator[str]:
        """
        Genera una respuesta en streaming basada en el contexto recuperado de LanceDB o,
        en su defecto, invoca al agente tras una frase introspectiva.
        
        Este método está diseñado para integración con servicios web que requieren streaming.
        
        Args:
            query: Pregunta del usuario
            historial_externo: Historial de conversación de la base de datos (opcional)
            
        Yields:
            str: Fragmentos de la respuesta en streaming
        """
        # Buscar contexto relevante en LanceDB según modo
        if self.mode == "hybrid":
            contexto = self._custom_hybrid_search(
                query=query, 
                similarity_threshold=self.similarity_threshold, 
                alpha=self.alpha
            )
        else:
            contexto = self._advanced_search(
                query=query,
                similarity_threshold=(self.adv_similarity_threshold or self.similarity_threshold),
                alpha=(self.adv_alpha if self.adv_alpha is not None else self.alpha)
            )
        
        # Inicializar respuesta
        respuesta = ""  

        if not contexto:    
            # Primero yield la introspección
            intro, intro_index = self._introspective_thinking()
            yield intro
            yield " "  # Espacio después de la introspección
            
            # Construcción del prompt con historial y contexto
            prompt = self._construir_prompt(query, contexto, historial_externo)
            
            # Ejecutar la consulta con RAG y yield cada fragmento
            response: Iterator = self.run(prompt, stream=True)
            if isinstance(response, Iterator):  # Si es un iterador, iteramos sobre él
                for resp in response:
                    fragmento = resp.content if resp.content else ""
                    respuesta += fragmento
                    yield fragmento
                yield "\n"  # Nueva línea al final
            else:
                respuesta = response.content if response.content else "No se generó respuesta."
                yield respuesta
        else:
            # Si hay contexto, yield cada línea del contexto
            lineas_contexto = contexto.split("\n")
            for linea in lineas_contexto:
                yield linea
                respuesta += linea
            yield "\n"  # Nueva línea al final
        
        # Actualizar historial después de completar la respuesta
        self._actualizar_historial(query, respuesta)

    def responder_con_rag_streaming_avanzado(self, query: str, historial_externo: str = None) -> Iterator[dict]:
        """
        Versión avanzada que devuelve un iterador de diccionarios con metadatos
        para mayor control en servicios web.
        
        Args:
            query: Pregunta del usuario
            historial_externo: Historial de conversación de la base de datos (opcional)
            
        Yields:
            dict: Diccionario con 'content', 'type' y 'metadata'
        """
        # Buscar contexto relevante en LanceDB con información de calidad según modo
        if self.mode == "hybrid":
            results_search = self._custom_hybrid_search_with_score(
                query=query, 
                similarity_threshold=self.similarity_threshold, 
                alpha=self.alpha
            )
        else:
            results_search = self._advanced_search_with_score(
                query=query,
                similarity_threshold=(self.adv_similarity_threshold or self.similarity_threshold),
                alpha=(self.adv_alpha if self.adv_alpha is not None else self.alpha)
            )
        
        contexto = results_search.get("contexto") if results_search else None
        best_score = results_search.get("best_score", 0) if results_search else 0
        
        # Inicializar respuesta
        respuesta = ""  

        if not contexto:    
            # Yield introspección con metadatos
            intro, intro_index = self._introspective_thinking()
            yield {
                "content": intro,
                "type": "introspection",
                "metadata": {"phase": "thinking", "reason": "no_context_found", "introspection_index": intro_index}
            }
            
            yield {
                "content": " ",
                "type": "separator",
                "metadata": {"phase": "thinking"}
            }
            
            # Construcción del prompt con historial y contexto
            prompt = self._construir_prompt(query, contexto, historial_externo)
            
            # Ejecutar la consulta con RAG y yield cada fragmento con metadatos
            response: Iterator = self.run(prompt, stream=True)
            if isinstance(response, Iterator):
                for resp in response:
                    fragmento = resp.content if resp.content else ""
                    respuesta += fragmento
                    yield {
                        "content": fragmento,
                        "type": "generation",
                        "metadata": {"phase": "generating", "model": self.model_id}
                    }
                
                yield {
                    "content": "\n",
                    "type": "separator",
                    "metadata": {"phase": "complete"}
                }
            else:
                respuesta = response.content if response.content else "No se generó respuesta."
                yield {
                    "content": respuesta,
                    "type": "generation",
                    "metadata": {"phase": "complete", "model": self.model_id}
                }
        else:
            # Si hay contexto, determinar si es respuesta directa o necesita streaming
            if best_score >= 0.8:
                # Respuesta directa de alta calidad - yield todo el contenido como un solo bloque
                yield {
                    "content": contexto,
                    "type": "direct_answer",
                    "metadata": {
                        "phase": "direct_response",
                        "source": "lancedb",
                        "confidence": best_score,
                        "reason": "high_confidence_match"
                    }
                }
                respuesta = contexto
            else:
                # Contexto de menor calidad - yield línea por línea como antes
                lineas_contexto = contexto.split("\n")
                for i, linea in enumerate(lineas_contexto):
                    yield {
                        "content": linea,
                        "type": "retrieval",
                        "metadata": {
                            "phase": "retrieving",
                            "source": "lancedb",
                            "confidence": best_score,
                            "line_index": i,
                            "total_lines": len(lineas_contexto)
                        }
                    }
                    respuesta += linea
                
                yield {
                    "content": "\n",
                    "type": "separator",
                    "metadata": {"phase": "complete"}
                }
        
        # Yield evento de finalización para indicar el fin del streaming
        yield {
            "content": "",
            "type": "completion",
            "metadata": {"phase": "finished", "total_response": respuesta, "confidence": best_score}
        }
        
        # Actualizar historial después de completar la respuesta
        self._actualizar_historial(query, respuesta)

    def responder_con_rag_streaming_simple(self, query: str) -> Iterator[str]:
        """
        Versión simplificada que devuelve solo el contenido sin metadatos.
        Útil para casos donde solo se necesita el texto en streaming.
        
        Args:
            query: Pregunta del usuario
            
        Yields:
            str: Fragmentos de la respuesta
        """
        for fragmento in self.responder_con_rag_streaming(query):
            yield fragmento

    def limpiar_historial(self):
        """Limpia el historial de conversación."""
        self.historial_conversacion = ""

    def obtener_historial(self) -> str:
        """Retorna el historial de conversación actual."""
        return self.historial_conversacion
    
    def establecer_historial_desde_bd(self, mensajes_bd) -> str:
        """
        Convierte mensajes de la base de datos al formato de historial interno.
        
        Args:
            mensajes_bd: Lista de mensajes de la BD con atributos role y message
            
        Returns:
            str: Historial formateado para usar en el contexto
        """
        if not mensajes_bd:
            return ""
            
        historial_partes = []
        for msg in mensajes_bd:
            if msg.role == "user":
                historial_partes.append(f"Usuario: {msg.message}")
            elif msg.role == "assistant":
                historial_partes.append(f"Qhali: {msg.message}")
        
        return self.separador_historial.join(historial_partes)

    def configurar_parametros_busqueda(
        self, 
        top_k_semantic: int = None,
        top_k_keyword: int = None,
        alpha: float = None,
        similarity_threshold: float = None
    ):
        """
        Permite configurar los parámetros de búsqueda dinámicamente.
        
        Args:
            top_k_semantic: Número de resultados semánticos
            top_k_keyword: Número de resultados por palabras clave
            alpha: Peso de la búsqueda semántica (0-1)
            similarity_threshold: Umbral de similitud mínimo
        """
        if top_k_semantic is not None:
            self.top_k_semantic = top_k_semantic
        if top_k_keyword is not None:
            self.top_k_keyword = top_k_keyword
        if alpha is not None:
            self.alpha = alpha
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
    
    def configurar_traduccion(self, enable_translation: bool = None):
        """
        Permite habilitar o deshabilitar la traducción dinámicamente.
        
        Args:
            enable_translation: Habilitar/deshabilitar traducción
        """
        if enable_translation is not None:
            self.enable_translation = enable_translation
            if enable_translation and not self.translator:
                print("⚠️ Traducción habilitada pero traductor no disponible")
    
    def obtener_estadisticas_traduccion(self):
        """
        Retorna estadísticas sobre el estado del traductor.
        
        Returns:
            dict: Información sobre el traductor
        """
        if not self.translator:
            return {
                "translation_enabled": False,
                "translator_available": False,
                "device": None,
                "compute_type": None
            }
        
        return {
            "translation_enabled": self.enable_translation,
            "translator_available": True,
            "device": self.translator.device,
            "compute_type": self.translator.compute_type,
            "model_dir": self.translator.model_dir
        }
    
    # ==================== Advanced RAG Methods ====================
    
    def configurar_modo(self, mode: str):
        """Permite alternar entre 'hybrid' y 'advanced' en caliente."""
        self.mode = mode if mode in {"hybrid", "advanced"} else "hybrid"
    
    def _expand_queries(self, query: str, n: int = None) -> List[str]:
        """Genera reformulaciones/paráfrasis breves de la query (multi-query)."""
        n = n or self.adv_num_queries
        prompt = dedent(f"""
        Genera {n} reformulaciones concisas de la siguiente consulta (sin listas numeradas, devuelve un JSON con una lista de strings):
        Consulta: "{query}"
        Responde SOLO con JSON de la forma: ["...", "...", ...]
        """).strip()
        try:
            resp = self.run(prompt, stream=False)
            text = resp.content if hasattr(resp, "content") else str(resp)
            import json, re
            m = re.search(r'\[.*\]', text, re.DOTALL)
            expansions = json.loads(m.group(0)) if m else []
            expansions = [x.strip() for x in expansions if isinstance(x, str) and x.strip()]
        except Exception:
            expansions = []
        # fallback: incluye la original si quedó vacío
        if not expansions:
            expansions = [query]
        # dedup conservador
        seen = set(); uniq=[]
        for q in expansions:
            if q.lower() not in seen:
                seen.add(q.lower()); uniq.append(q)
        return uniq[:n]
    
    def _retrieve_for_single_query(self, q: str, alpha: float, top_k_sem: int, top_k_kw: int) -> pl.DataFrame:
        """Reutiliza la canalización híbrida para una subconsulta (sin combinar aún)."""
        search_q = q
        if self.enable_translation and self.translator:
            try:
                search_q = self.translator.translate_es_to_en(q)
            except Exception:
                search_q = q
        qvec = np.array(self.embedder.get_embedding(search_q), dtype=np.float32) if alpha > 0 else None
        r_sem = self._search_semantic(qvec, top_k=top_k_sem) if alpha > 0 else pl.DataFrame()
        r_kw  = self._search_keyword(search_q, top_k=top_k_kw) if alpha < 1 else pl.DataFrame()
        if r_sem.is_empty() and r_kw.is_empty():
            return pl.DataFrame()
        combined = self._combine_results(r_sem, r_kw, alpha)
        return combined if combined is not None else pl.DataFrame()
    
    def _merge_across_queries(self, list_of_dfs: List[pl.DataFrame]) -> pl.DataFrame:
        """Une resultados de múltiples subconsultas y de-duplica por chunk."""
        if not list_of_dfs:
            return pl.DataFrame()
        # concat ligera (algunas tablas podrían estar vacías)
        dfs = [df for df in list_of_dfs if df is not None and not df.is_empty()]
        if not dfs:
            return pl.DataFrame()
        df = pl.concat(dfs, how="vertical", rechunk=True)

        # Asegurar chunk_key para deduplicación
        if "chunk_key" not in df.columns and {"pdf_name","chunk_index"}.issubset(set(df.columns)):
            df = df.with_columns((pl.col("pdf_name").cast(pl.String) + "_" + pl.col("chunk_index").cast(pl.String)).alias("chunk_key"))

        # Agrega un voto por aparición de chunk en diferentes subconsultas
        grouped = (
            df.group_by("chunk_key")
              .agg([
                  pl.first("pdf_name").alias("pdf_name"),
                  pl.first("chunk_index").alias("chunk_index"),
                  pl.first("text").alias("text"),
                  pl.coalesce([pl.max("titulo"), pl.lit("")]).alias("titulo"),
                  pl.coalesce([pl.max("fuente"), pl.lit("")]).alias("fuente"),
                  pl.max("score_final").alias("score_final_max"),
                  pl.mean("score_final").alias("score_final_mean"),
                  pl.len().alias("votes")
              ])
        )
        return grouped
    
    def _rerank_and_diversify(self, df: pl.DataFrame, k: int) -> pl.DataFrame:
        """Reordena priorizando evidencia frecuente y diversidad (MMR simple)."""
        if df.is_empty():
            return df
        # score compuesto base
        if self.adv_merge_strategy == "vote":
            base = df.with_columns((pl.col("votes").cast(pl.Float64)).alias("_base"))
        elif self.adv_merge_strategy == "avg":
            base = df.with_columns(pl.col("score_final_mean").alias("_base"))
        elif self.adv_merge_strategy == "max":
            base = df.with_columns(pl.col("score_final_max").alias("_base"))
        else:  # "vote+score"
            base = df.with_columns((pl.col("score_final_mean") + 0.1 * pl.col("votes").cast(pl.Float64)).alias("_base"))

        base = base.sort("_base", descending=True)

        if self.adv_rerank_strategy != "mmr":
            return base.head(k)

        # MMR (simple): usa embeddings del texto para penalizar similitud con seleccionados
        try:
            import math
            selected = []
            cand = base.to_dicts()
            # precalcula embeddings
            def emb(t): return np.array(self.embedder.get_embedding(t or ""), dtype=np.float32)
            embs = [emb(row["text"]) for row in cand]

            lam = 0.7  # tradeoff relevancia/diversidad
            chosen = set()
            while len(selected) < min(k, len(cand)):
                best_i, best_score = None, -1e9
                for i, row in enumerate(cand):
                    if i in chosen: 
                        continue
                    rel = row["_base"]
                    div = 0.0
                    if selected:
                        sims = []
                        for j in selected:
                            # cos sim
                            a,b = embs[i], embs[j]
                            denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)
                            sims.append(float(np.dot(a,b)/denom))
                        div = max(sims) if sims else 0.0
                    mmr = lam*rel - (1-lam)*div
                    if mmr > best_score:
                        best_score, best_i = mmr, i
                selected.append(best_i); chosen.add(best_i)
            out = [cand[i] for i in selected]
            return pl.DataFrame(out)
        except Exception:
            return base.head(k)
    
    def _advanced_search(self, query: str, similarity_threshold: float = None, alpha: float = None):
        """Canalización Advanced: expandir → recuperar por subquery → unir/deduplicar → rerank → traducir → contexto."""
        alpha = self.alpha if alpha is None else alpha
        similarity_threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold

        subs = self._expand_queries(query, n=self.adv_num_queries)
        dfs = []
        for sub in subs:
            dfs.append(self._retrieve_for_single_query(sub, alpha, self.adv_top_k_per_query, self.adv_top_k_per_query))

        merged = self._merge_across_queries(dfs)
        if merged.is_empty():
            return None

        # filtra por umbral sobre score promedio
        merged = merged.with_columns(pl.col("score_final_mean").alias("_sf"))
        merged = merged.filter(pl.col("_sf") >= (self.adv_similarity_threshold or similarity_threshold))
        if merged.is_empty():
            return None

        ranked = self._rerank_and_diversify(merged, k=self.adv_max_chunks)
        if ranked.is_empty():
            return None

        # Traducción de chunks EN->ES (reutiliza tu pipeline)
        contexto_chunks = []
        for row in ranked.to_dicts():
            chunk_text = row["text"]
            if self.enable_translation and self.translator and not self.skip_chunk_translation:
                try:
                    chunk_text = self.translator.translate_en_to_es(chunk_text)
                except Exception:
                    pass
            titulo = row.get("titulo", "")
            pdf_name = row.get("pdf_name", "documento")
            if titulo:
                contexto_chunks.append(f"**{titulo}** ({pdf_name}):\n{chunk_text}")
            else:
                contexto_chunks.append(f"**{pdf_name}**:\n{chunk_text}")

        if not contexto_chunks:
            return None
        if len(contexto_chunks) == 1 and ranked["score_final_mean"][0] >= 0.8:
            return contexto_chunks[0]
        return "\n\n".join(contexto_chunks)
    
    def _advanced_search_with_score(self, query: str, similarity_threshold: float = None, alpha: float = None):
        """Igual que _advanced_search pero retornando best_score para control de UI."""
        alpha = self.alpha if alpha is None else alpha
        similarity_threshold = self.similarity_threshold if similarity_threshold is None else similarity_threshold

        subs = self._expand_queries(query, n=self.adv_num_queries)
        dfs = [self._retrieve_for_single_query(s, alpha, self.adv_top_k_per_query, self.adv_top_k_per_query) for s in subs]
        merged = self._merge_across_queries(dfs)
        if merged.is_empty():
            return None

        merged = merged.with_columns(pl.col("score_final_mean").alias("_sf"))
        merged = merged.filter(pl.col("_sf") >= (self.adv_similarity_threshold or similarity_threshold))
        if merged.is_empty():
            return None

        ranked = self._rerank_and_diversify(merged, k=self.adv_max_chunks)
        if ranked.is_empty():
            return None

        best_score = float(ranked["score_final_mean"][0]) if "score_final_mean" in ranked.columns else 0.0
        contexto = self._advanced_search(query, similarity_threshold=similarity_threshold, alpha=alpha)
        if not contexto:
            return None
        return {"contexto": contexto, "best_score": best_score}
    
    def retrieve_only_advanced(self, query: str, top_k_per_query: int = None, alpha: float = None):
        """Versión retrieve_only para Advanced: devuelve lista con metadatos deduplicados."""
        alpha = self.alpha if alpha is None else alpha
        top_k_per_query = top_k_per_query or self.adv_top_k_per_query
        subs = self._expand_queries(query, n=self.adv_num_queries)
        dfs = [self._retrieve_for_single_query(s, alpha, top_k_per_query, top_k_per_query) for s in subs]
        merged = self._merge_across_queries(dfs)
        if merged.is_empty():
            return []
        ranked = self._rerank_and_diversify(merged, k=self.adv_max_chunks)
        if ranked.is_empty():
            return []
        # genera salida compatible
        out = []
        for row in ranked.to_dicts():
            out.append({
                "pdf_name": row.get("pdf_name", ""),
                "chunk_index": row.get("chunk_index", 0),
                "chunk_id": f"{row.get('pdf_name','')}#{row.get('chunk_index',0)}",
                "text": row.get("text", ""),
                "score_final": float(row.get("score_final_mean", row.get("score_final_max", 0.0))),
                "titulo": row.get("titulo", ""),
                "fuente": row.get("fuente", "")
            })
        return out
