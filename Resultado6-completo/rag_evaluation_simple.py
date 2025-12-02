#!/usr/bin/env python3
"""
Sistema de evaluación RAG simplificado con métricas validadas por expertos.
Incluye Hit@K, MRR, nDCG@K para retrieval y EXACT MATCH, F1 Score, 
Similitud Semántica para generación siguiendo buenas prácticas de utils_custom_metrics.py.

Detecta automáticamente el tipo de modelo (base, dpo, orpo, finetuned) según el nombre
y genera reportes compatibles con golden_f1_results.csv de qlora_pref_train.py.
"""

import sys
import time
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importar métricas personalizadas
try:
    from utils_custom_metrics import (
        hit_at_k, mean_reciprocal_rank, ndcg_at_k, 
        normalize_text, calculate_retrieval_metrics,
        calculate_exact_match, calculate_f1_score, calculate_semantic_similarity
    )
    print("✅ Métricas personalizadas importadas correctamente")
except ImportError as e:
    print(f"❌ Error importando métricas personalizadas: {e}")
    sys.exit(1)

# Importar RAGAgent
try:
    from rag_agent import RAGAgent
except ImportError as e:
    print(f"❌ Error importando RAGAgent: {e}")
    sys.exit(1)


class SimpleRAGEvaluationSystem:
    """
    Sistema simplificado de evaluación RAG con métricas clásicas de retrieval.
    Detecta automáticamente el tipo de modelo (base, dpo, orpo) y reporta adecuadamente.
    """
    
    def __init__(
        self,
        agent: RAGAgent,
        golden_set_file: str = "golden_set_test.csv",
        results_dir: str = "evaluation_results",
        generate_answers: bool = False,  # False para rapidez, no calcula generación
        verbose: bool = True,
        model_id: str = None,  # Identificador del modelo para detección de tipo
        rag_mode: str = "hybrid",  # Modo de RAG: "hybrid" o "advanced"
        use_translation: bool = False  # Si usar traducción o no
    ):
        """
        Inicializar el sistema de evaluación.
        
        Args:
            agent: Instancia del RAGAgent a evaluar
            golden_set_file: Archivo CSV con el conjunto golden
            results_dir: Directorio para guardar resultados
            generate_answers: Si generar respuestas (costoso computacionalmente)
            verbose: Si mostrar información detallada
        """
        self.agent = agent
        self.golden_set_file = golden_set_file
        self.results_dir = Path(results_dir)
        self.generate_answers = generate_answers
        self.verbose = verbose
        self.model_id = model_id or "unknown"
        self.rag_mode = rag_mode  # "hybrid" o "advanced"
        self.use_translation = use_translation
        
        # Detectar tipo de modelo y método
        self.run_type, self.method = self._detect_model_type(self.model_id)
        
        # Crear directorio de resultados
        self.results_dir.mkdir(exist_ok=True)
        
        # Cargar conjunto golden y detectar capacidades
        self.golden_data, self.has_chunk_ids = self._load_golden_set()
        
        # Configuraciones de evaluación
        self.alpha_values = [i / 20.0 for i in range(21)]  # 0.0 to 1.0 con saltos de 0.05
        self.k_values = [1, 2, 3, 5, 10]  # Hit@1, Hit@2, Hit@3, Hit@5, Hit@10
        
        print(f"✅ Sistema de evaluación inicializado")
        print(f"   • Modelo ID: {self.model_id}")
        print(f"   • Tipo detectado: {self.run_type}")
        print(f"   • Método: {self.method}")
        print(f"   • Modo RAG: {self.rag_mode}")
        print(f"   • Traducción: {'✅ Habilitada' if self.use_translation else '❌ Deshabilitada'}")
        print(f"   • Conjunto golden: {len(self.golden_data)} preguntas")
        print(f"   • Chunk-level evaluation: {'✅ Disponible' if self.has_chunk_ids else '❌ Solo document-level'}")
        print(f"   • Generar respuestas: {generate_answers}")
    
    def _detect_model_type(self, model_id: str, verbose: bool = None) -> Tuple[str, str]:
        """
        Detecta el tipo de modelo y método de entrenamiento según el nombre.
        
        Args:
            model_id: Identificador del modelo
            verbose: Si mostrar información de depuración (usa self.verbose si no se especifica)
            
        Returns:
            tuple: (run_type, method)
                run_type: 'base', 'finetuned', 'dpo', 'orpo', 'base-ollama', 'ft-ollama'
                method: 'dpo', 'orpo', '-' (guion para base sin fine-tuning)
        """
        if verbose is None:
            verbose = getattr(self, 'verbose', False)
            
        model_lower = model_id.lower()
        
        # Detectar método (dpo o orpo)
        method = "-"  # Por defecto: modelo base sin método
        if "dpo" in model_lower:
            method = "dpo"
        elif "orpo" in model_lower:
            method = "orpo"
        
        # Detectar run_type
        run_type = "base-ollama"
        
        # Modelos finetuned típicos
        if any(x in model_lower for x in ["qlora", "lora", "peft", "fine-tuned", "finetuned", "ft-"]):
            if method != "-":
                run_type = "ft-ollama"                
                
        



        if verbose:
            print(f"🔍 Detección de modelo: '{model_id}' → run_type='{run_type}', method='{method}'")
        
        return run_type, method
    
    def _load_golden_set(self) -> Tuple[List[Dict], bool]:
        """
        Cargar y validar el conjunto golden desde CSV, detectando dinámicamente
        si tiene gold_chunk_ids disponibles.
        
        Returns:
            tuple: (lista de diccionarios con datos golden, bool indicando si hay chunk_ids)
        """
        try:
            df = pl.read_csv(self.golden_set_file, encoding='utf-8')
            
            # Validar columnas requeridas
            required_cols = ['qid', 'question']
            
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Columna requerida '{col}' no encontrada en {self.golden_set_file}")
            
            # Detectar si tiene gold_chunk_ids
            has_chunk_ids = 'gold_chunk_ids' in df.columns
            
            # Convertir a lista de diccionarios y procesar
            golden_data = []
            for row in df.to_dicts():
                item = {
                    'qid': str(row['qid']),
                    'question': str(row['question']),
                    'reference_answer': row.get('reference_answer', ''),
                    'gold_doc_ids': [],
                    'gold_chunk_ids': []
                }
                
                # Procesar gold_doc_ids (separados por |) - SIEMPRE requerido
                if row.get('gold_doc_ids'):
                    item['gold_doc_ids'] = [
                        doc_id.strip() 
                        for doc_id in str(row['gold_doc_ids']).split('|')
                        if doc_id.strip()
                    ]
                
                # Procesar gold_chunk_ids (separados por |) - SOLO si existe la columna
                if has_chunk_ids and row.get('gold_chunk_ids'):
                    item['gold_chunk_ids'] = [
                        chunk_id.strip()
                        for chunk_id in str(row['gold_chunk_ids']).split('|')
                        if chunk_id.strip()
                    ]
                
                golden_data.append(item)
            
            if self.verbose:
                print(f"✅ Conjunto golden cargado: {len(golden_data)} entradas")
                sample = golden_data[0] if golden_data else {}
                print(f"   • Ejemplo qid: {sample.get('qid', 'N/A')}")
                print(f"   • Gold docs: {len(sample.get('gold_doc_ids', []))} documentos")
                if has_chunk_ids:
                    print(f"   • Gold chunks: {len(sample.get('gold_chunk_ids', []))} chunks")
                else:
                    print(f"   • Gold chunks: N/A (columna no disponible)")
            
            return golden_data, has_chunk_ids
            
        except Exception as e:
            print(f"❌ Error cargando conjunto golden: {e}")
            sys.exit(1)
    
    def _calculate_dynamic_retrieval_metrics(
        self,
        retrieved_doc_ids: List[str],
        gold_doc_ids: List[str],
        retrieved_chunk_ids: List[str],
        gold_chunk_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calcular métricas de retrieval dinámicamente según disponibilidad de gold_chunk_ids.
        
        Modo A (sin gold_chunk_ids): Solo métricas a nivel documento
        Modo B (con gold_chunk_ids): Métricas a nivel documento Y chunk
        
        Args:
            retrieved_doc_ids: IDs de documentos recuperados
            gold_doc_ids: IDs de documentos ground truth
            retrieved_chunk_ids: IDs de chunks recuperados
            gold_chunk_ids: IDs de chunks ground truth (vacío si no disponible)
            
        Returns:
            dict: Diccionario con métricas calculadas según disponibilidad
        """
        metrics = {}
        
        # SIEMPRE calcular métricas a nivel documento
        for k in self.k_values:
            doc_hit_k = hit_at_k(retrieved_doc_ids, gold_doc_ids, k)
            metrics[f"document_hit_at_{k}"] = doc_hit_k
        
        # MRR y nDCG a nivel documento
        metrics["document_mrr"] = mean_reciprocal_rank(retrieved_doc_ids, gold_doc_ids)
        metrics["document_ndcg_at_5"] = ndcg_at_k(retrieved_doc_ids, gold_doc_ids, k=5)
        metrics["document_ndcg_at_10"] = ndcg_at_k(retrieved_doc_ids, gold_doc_ids, k=10)
        
        # SOLO calcular métricas a nivel chunk si hay gold_chunk_ids disponibles
        if self.has_chunk_ids and gold_chunk_ids:
            if self.verbose:
                # Solo mostrar este mensaje una vez por configuración
                pass  # Comentario silencioso
            
            for k in self.k_values:
                chunk_hit_k = hit_at_k(retrieved_chunk_ids, gold_chunk_ids, k)
                metrics[f"chunk_hit_at_{k}"] = chunk_hit_k
            
            # MRR y nDCG a nivel chunk
            metrics["chunk_mrr"] = mean_reciprocal_rank(retrieved_chunk_ids, gold_chunk_ids)
            metrics["chunk_ndcg_at_5"] = ndcg_at_k(retrieved_chunk_ids, gold_chunk_ids, k=5)
            metrics["chunk_ndcg_at_10"] = ndcg_at_k(retrieved_chunk_ids, gold_chunk_ids, k=10)
        else:
            # Si no hay chunk IDs disponibles, establecer métricas chunk en 0.0
            for k in self.k_values:
                metrics[f"chunk_hit_at_{k}"] = 0.0
            
            metrics["chunk_mrr"] = 0.0
            metrics["chunk_ndcg_at_5"] = 0.0
            metrics["chunk_ndcg_at_10"] = 0.0
        
        return metrics
    
    def evaluate_single_question(
        self,
        question: str,
        gold_item: Dict,
        alpha: float,
        top_k_retrieval: int = 50
    ) -> Dict[str, Any]:
        """
        Evaluar una sola pregunta con métricas completas.
        
        Args:
            question: Pregunta a evaluar
            gold_item: Item del conjunto golden con ground truth
            alpha: Parámetro alpha para búsqueda híbrida
            top_k_retrieval: Número de documentos a recuperar
            
        Returns:
            dict: Diccionario con todos los resultados de evaluación
        """
        start_time = time.time()
        
        # PASO 1: Recuperación según el modo de RAG
        if self.rag_mode == "advanced":
            # Modo Advanced: usa retrieve_only_advanced con sus propios parámetros
            # (adv_num_queries, adv_top_k_per_query) para acotar el coste
            # NO pasamos top_k_per_query aquí para usar el configurado en el agente
            retrieved_chunks = self.agent.retrieve_only_advanced(
                query=question,
                alpha=alpha
            )
        else:
            # Modo Hybrid (default): usa retrieve_only
            retrieved_chunks = self.agent.retrieve_only(
                query=question,
                top_k_semantic=top_k_retrieval,
                top_k_keyword=top_k_retrieval,
                alpha=alpha
            )
        
        retrieval_time = time.time() - start_time
        
        # Extraer IDs para evaluación
        retrieved_doc_ids = [chunk["pdf_name"] for chunk in retrieved_chunks]
        retrieved_chunk_ids = [chunk["chunk_id"] for chunk in retrieved_chunks]
        
        # PASO 2: Generar respuesta (si está habilitado) usando pipeline completo con traducción
        generated_answer = ""
        generation_time = 0.0
        
        if self.generate_answers and retrieved_chunks:
            gen_start = time.time()
            try:
                contexto_es = ""
                best_score = 0.0

                if self.rag_mode == "advanced":
                    # ⚡ ADVANCED: reutilizar la recuperación avanzada ya hecha
                    # (retrieve_only_advanced) y NO volver a invocar _advanced_search_with_score.
                    # Usar adv_max_chunks del agente si está configurado
                    max_chunks_limit = getattr(self.agent, "adv_max_chunks", 5)
                    max_chunks = min(len(retrieved_chunks), max_chunks_limit)
                    top_chunks = retrieved_chunks[:max_chunks]

                    if top_chunks:
                        # Usamos el score del mejor chunk recuperado como proxy de confianza
                        best_score = top_chunks[0].get("score_final", 0.0)

                        use_translation = (
                            getattr(self.agent, "enable_translation", False)
                            and not getattr(self.agent, "skip_chunk_translation", False)
                            and getattr(self.agent, "translator", None) is not None
                        )

                        formatted_chunks = []
                        for ch in top_chunks:
                            text = ch.get("text", "")
                            if use_translation:
                                try:
                                    text = self.agent.translator.translate_en_to_es(text)
                                except Exception as te:
                                    if self.verbose:
                                        print(f"⚠️ Error traduciendo chunk: {te}")
                                    # Fallback: usar texto original

                            titulo = ch.get("titulo") or ch.get("pdf_name") or "Fragmento"
                            pdf_name = ch.get("pdf_name") or ""
                            header = f"**{titulo}**"
                            if pdf_name and pdf_name not in header:
                                header += f" ({pdf_name})"
                            formatted_chunks.append(f"{header}:\n{text}")

                        contexto_es = "\n\n".join(formatted_chunks)

                else:
                    # MODO HYBRID: mantenemos el pipeline existente que ya maneja traducción
                    response_result = self.agent._custom_hybrid_search_with_score(
                        query=question,
                        top_k_semantic=5,  # Limitar para generación
                        top_k_keyword=5,
                        alpha=alpha,
                        similarity_threshold=0.0  # Sin filtro para evaluación
                    )
                    if response_result and response_result.get("contexto"):
                        contexto_es = response_result["contexto"]
                        best_score = response_result.get("best_score", 0.0)

                # Si no se logró construir contexto_es (p.ej., sin chunks), usar fallback simple
                if not contexto_es:
                    context_chunks = retrieved_chunks[:5]
                    contexto_es = "\n\n".join(
                        f"**{chunk.get('titulo', chunk['pdf_name'])}**:\n{chunk['text']}"
                        for chunk in context_chunks
                    )
                    if not best_score and retrieved_chunks:
                        best_score = retrieved_chunks[0].get("score_final", 0.0)

                # RECOMENDACIÓN DEL EXPERTO: modo extractivo cuando el score es alto
                if best_score >= 0.8:
                    from utils_custom_metrics import extract_best_answer_from_context

                    extracted_answer = extract_best_answer_from_context(question, contexto_es)
                    if extracted_answer:
                        generated_answer = extracted_answer
                    else:
                        generated_answer = self.agent.generate_answer_for_evaluation(
                            query=question,
                            contexto=contexto_es,
                            use_json_format=False
                        )
                else:
                    generated_answer = self.agent.generate_answer_for_evaluation(
                        query=question,
                        contexto=contexto_es,
                        use_json_format=False
                    )

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error generando respuesta: {e}")
                generated_answer = ""

            generation_time = time.time() - gen_start
        
        # PASO 3: Calcular métricas de retrieval según disponibilidad
        retrieval_metrics = self._calculate_dynamic_retrieval_metrics(
            retrieved_doc_ids=retrieved_doc_ids,
            gold_doc_ids=gold_item["gold_doc_ids"],
            retrieved_chunk_ids=retrieved_chunk_ids,
            gold_chunk_ids=gold_item["gold_chunk_ids"] if self.has_chunk_ids else []
        )
        
        # PASO 4: Calcular métricas de generación si hay respuesta de referencia
        exact_match = 0.0
        f1_score = 0.0
        semantic_similarity = 0.0
        
        if gold_item.get("reference_answer") and self.generate_answers:
            # MEJORAS IMPLEMENTADAS SEGÚN RECOMENDACIONES DEL EXPERTO:
            # 1. Normalización ampliada para números y espacios
            # 2. Formato controlado en generación para maximizar EM
            # 3. Extracción directa del contexto para respuestas de alta calidad
            
            exact_match = calculate_exact_match(generated_answer, gold_item["reference_answer"])
            f1_score = calculate_f1_score(generated_answer, gold_item["reference_answer"])
            
            # Calcular similitud semántica usando el embedder del agente
            try:
                semantic_similarity = calculate_semantic_similarity(
                    generated_answer, 
                    gold_item["reference_answer"], 
                    embedder=self.agent.embedder
                )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Error calculando similitud semántica: {e}")
                semantic_similarity = calculate_semantic_similarity(
                    generated_answer, 
                    gold_item["reference_answer"], 
                    embedder=None  # Usar fallback simple
                )
                
            # DEBUG: Mostrar comparación para casos de EM = 0 (opcional)
            if self.verbose and exact_match == 0.0 and generated_answer and gold_item["reference_answer"]:
                from utils_custom_metrics import normalize_text
                gen_norm = normalize_text(generated_answer)
                ref_norm = normalize_text(gold_item["reference_answer"])
                print(f"   🔍 DEBUG EM=0: '{gen_norm[:50]}...' vs '{ref_norm[:50]}...'")
                print(f"       F1={f1_score:.3f}, Sem={semantic_similarity:.3f}")
        
        # Compilar resultados - Compatible con golden_f1_results.csv
        results = {
            # Metadatos principales (compatibles con qlora_pref_train.py)
            "qid": gold_item["qid"],
            "question": question,
            "reference_answer": gold_item["reference_answer"],
            "generated_answer": generated_answer,
            "f1_score": f1_score,
            "model_id": self.model_id,
            "run_type": self.run_type,
            "method": self.method,
            "rag_mode": self.rag_mode,  # Añadido: tipo de RAG usado
            "use_translation": self.use_translation,  # Añadido: si se usó traducción
            "dataset": Path(self.golden_set_file).name,
            "eval_timestamp": datetime.now().isoformat(timespec="seconds"),
            
            # Parámetros de evaluación
            "alpha": alpha,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "total_time": retrieval_time + generation_time,
            
            # Información de recuperación
            "num_retrieved": len(retrieved_chunks),
            "best_score": retrieved_chunks[0]["score_final"] if retrieved_chunks else 0.0,
            "avg_score": np.mean([c["score_final"] for c in retrieved_chunks]) if retrieved_chunks else 0.0,
            
            # Métricas clásicas de retrieval
            **retrieval_metrics,
            
            # Métricas de generación adicionales
            "exact_match": exact_match,
            "semantic_similarity": semantic_similarity,
            
            # IDs para análisis posterior
            "retrieved_doc_ids": retrieved_doc_ids[:10],  # Limitar para CSV
            "retrieved_chunk_ids": retrieved_chunk_ids[:10],
            "gold_doc_ids": gold_item["gold_doc_ids"],
            "gold_chunk_ids": gold_item["gold_chunk_ids"]
        }
        
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """
        Ejecutar evaluación completa con todas las configuraciones alpha.
        
        Returns:
            dict: Resultados completos de evaluación
        """
        print("🚀 INICIANDO EVALUACIÓN RAG SIMPLIFICADA")
        print("=" * 80)
        
        all_results = []
        
        # Evaluar para cada alpha
        for alpha in self.alpha_values:
            alpha_name = self._get_alpha_name(alpha)
            print(f"\n🔧 Evaluando configuración: {alpha_name} (α={alpha:.1f})")
            print("-" * 60)
            
            alpha_results = []
            
            # Evaluar cada pregunta
            for i, gold_item in enumerate(self.golden_data, 1):
                question = gold_item["question"]
                qid = gold_item["qid"]
                
                if self.verbose:
                    print(f"   📋 [{i}/{len(self.golden_data)}] Evaluando qid={qid}")
                    print(f"        {question[:60]}...")
                
                # Evaluar pregunta individual
                result = self.evaluate_single_question(
                    question=question,
                    gold_item=gold_item,
                    alpha=alpha,
                    top_k_retrieval=50
                )
                
                alpha_results.append(result)
                all_results.append(result)
                
                if self.verbose:
                    print(f"        ✓ Doc Hit@1: {result.get('document_hit_at_1', 0):.3f} | "
                          f"Doc Hit@2: {result.get('document_hit_at_2', 0):.3f} | "
                          f"Doc Hit@3: {result.get('document_hit_at_3', 0):.3f} | "
                          f"Doc Hit@5: {result.get('document_hit_at_5', 0):.3f} | "
                          f"Doc Hit@10: {result.get('document_hit_at_10', 0):.3f} | "
                          f"Chunk Hit@1: {result.get('chunk_hit_at_1', 0):.3f} | "
                          f"Chunk Hit@2: {result.get('chunk_hit_at_2', 0):.3f} | "
                          f"Chunk Hit@3: {result.get('chunk_hit_at_3', 0):.3f} | "
                          f"Chunk Hit@5: {result.get('chunk_hit_at_5', 0):.3f} | "
                          f"Chunk Hit@10: {result.get('chunk_hit_at_10', 0):.3f} | "
                          f"Best Score: {result['best_score']:.3f}")
            
            # Estadísticas por alpha
            self._print_alpha_stats(alpha_name, alpha_results)
        
        # Guardar resultados detallados
        print(f"\n💾 Guardando resultados...")
        self._save_detailed_results(all_results)
        
        # Generar análisis
        print(f"\n📊 Generando análisis...")
        analysis = self._generate_analysis(all_results)
        
        # GRÁFICOS COMENTADOS: Descomentar si se necesitan visualizaciones
        # print(f"\n📊 Generando gráficos...")
        # self._create_visualizations(all_results)
        
        # Resumen final
        self._print_final_summary(analysis)
        
        return {
            "detailed_results": all_results,
            "analysis": analysis
        }
    
    def _get_alpha_name(self, alpha: float) -> str:
        """Obtener nombre descriptivo para valor alpha."""
        if alpha == 0.0:
            return "Solo Keywords (BM25)"
        elif alpha == 1.0:
            return "Solo Semántico"
        else:
            sem_percent = int(alpha * 100)
            kw_percent = 100 - sem_percent
            return f"Híbrido ({sem_percent}% Sem + {kw_percent}% BM25)"
    
    def _print_alpha_stats(self, alpha_name: str, results: List[Dict]):
        """
        Imprimir estadísticas para una configuración alpha, adaptándose dinámicamente.
        
        LÓGICA DE MÉTRICAS:
        - Document Hit@1: Precisión en la primera posición (más estricta)
        - Document Hit@2: Precisión en las primeras 2 posiciones (intermedia)
        - Document Hit@3: Precisión en las primeras 3 posiciones (más permisiva)
        - MRR: Calidad del ranking considerando la posición del primer documento correcto
        - Best Score: Confianza promedio del sistema en sus resultados
        """
        if not results:
            return
        
        # Métricas de retrieval - documento (SIEMPRE disponible)
        # LÓGICA: Calcular promedios de las métricas de precisión por posición
        doc_hit1 = np.mean([r.get('document_hit_at_1', 0) for r in results])
        doc_hit2 = np.mean([r.get('document_hit_at_2', 0) for r in results])
        doc_hit3 = np.mean([r.get('document_hit_at_3', 0) for r in results])
        doc_hit5 = np.mean([r.get('document_hit_at_5', 0) for r in results])
        doc_hit10 = np.mean([r.get('document_hit_at_10', 0) for r in results])
        doc_mrr = np.mean([r.get('document_mrr', 0) for r in results])
        avg_score = np.mean([r['best_score'] for r in results])
        
        # Métricas de retrieval - chunk (SOLO si disponible)
        # LÓGICA: Solo calcular si hay ground truth a nivel de chunk disponible
        chunk_stats = ""
        if self.has_chunk_ids:
            chunk_hit1 = np.mean([r.get('chunk_hit_at_1', 0) for r in results])
            chunk_hit2 = np.mean([r.get('chunk_hit_at_2', 0) for r in results])
            chunk_hit3 = np.mean([r.get('chunk_hit_at_3', 0) for r in results])
            chunk_hit5 = np.mean([r.get('chunk_hit_at_5', 0) for r in results])
            chunk_hit10 = np.mean([r.get('chunk_hit_at_10', 0) for r in results])
            chunk_mrr = np.mean([r.get('chunk_mrr', 0) for r in results])
            chunk_stats = f" | Chunk Hit@1: {chunk_hit1:.3f} | Chunk Hit@2: {chunk_hit2:.3f} | Chunk Hit@3: {chunk_hit3:.3f} | Chunk Hit@5: {chunk_hit5:.3f} | Chunk Hit@10: {chunk_hit10:.3f} | Chunk MRR: {chunk_mrr:.3f}"
        
        # Métricas de generación (si están habilitadas) - F1 y Semantic Similarity
        generation_stats = ""
        if self.generate_answers:
            f1_score = np.mean([r.get('f1_score', 0) for r in results])
            semantic_sim = np.mean([r.get('semantic_similarity', 0) for r in results])
            generation_stats = f" | F1: {f1_score:.3f} | Sem: {semantic_sim:.3f}"
        
        # Rendimiento
        avg_time = np.mean([r['total_time'] for r in results])
        
        print(f"      📈 Estadísticas {alpha_name}:")
        print(f"         🔍 RETRIEVAL: Doc Hit@1: {doc_hit1:.3f} | Doc Hit@2: {doc_hit2:.3f} | Doc Hit@3: {doc_hit3:.3f} | Doc Hit@5: {doc_hit5:.3f} | Doc Hit@10: {doc_hit10:.3f} | Doc MRR: {doc_mrr:.3f}{chunk_stats}")
        print(f"         ⚡ RENDIMIENTO: Avg Score: {avg_score:.3f}{generation_stats} | Tiempo: {avg_time:.2f}s")
    
    def _save_detailed_results(self, results: List[Dict]):
        """Guardar resultados en archivo único CSV en modo append."""
        try:
            # Simplificar listas para CSV
            simplified_results = []
            for result in results:
                simplified = result.copy()
                # Convertir listas a strings para CSV
                if isinstance(simplified.get("retrieved_doc_ids"), list):
                    simplified["retrieved_doc_ids"] = "|".join(simplified["retrieved_doc_ids"])
                if isinstance(simplified.get("retrieved_chunk_ids"), list):
                    simplified["retrieved_chunk_ids"] = "|".join(simplified["retrieved_chunk_ids"])
                if isinstance(simplified.get("gold_doc_ids"), list):
                    simplified["gold_doc_ids"] = "|".join(simplified["gold_doc_ids"])
                if isinstance(simplified.get("gold_chunk_ids"), list):
                    simplified["gold_chunk_ids"] = "|".join(simplified["gold_chunk_ids"])
                simplified_results.append(simplified)
            
            # Convertir a DataFrame
            df = pl.DataFrame(simplified_results)
            
            # ARCHIVO ÚNICO: rag_evaluation_results.csv en evaluation_results/
            output_file = Path("evaluation_results") / "rag_evaluation_results.csv"
            output_file.parent.mkdir(exist_ok=True)
            
            # Modo APPEND: agregar a archivo existente o crear nuevo
            if output_file.exists():
                try:
                    existing_df = pl.read_csv(str(output_file), infer_schema_length=10000)
                    # Concatenar usando vertical_relaxed para manejar esquemas ligeramente diferentes
                    combined_df = pl.concat([existing_df, df], how="vertical_relaxed")
                    combined_df.write_csv(str(output_file))
                    print(f"✅ Resultados agregados (APPEND) a: {output_file}")
                    print(f"   Total de registros: {len(combined_df)}")
                except Exception as e:
                    print(f"⚠️ No se pudo hacer APPEND, guardando archivo nuevo: {e}")
                    df.write_csv(str(output_file))
                    print(f"✅ Resultados guardados (sobrescrito): {output_file}")
            else:
                df.write_csv(str(output_file))
                print(f"✅ Resultados guardados (nuevo archivo): {output_file}")
                print(f"   Total de registros: {len(df)}")
                
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
    
    def _generate_analysis(self, results: List[Dict]) -> Dict:
        """
        Generar análisis estadístico de resultados.
        
        LÓGICA DEL ANÁLISIS:
        1. Agrupar resultados por valor de alpha (0.0 a 1.0)
        2. Calcular estadísticas promedio para cada métrica por alpha
        3. Identificar la mejor configuración alpha para cada métrica
        4. Proporcionar recomendaciones basadas en el rendimiento
        """
        df = pl.DataFrame(results)
        
        # Análisis por alpha: Calcular estadísticas para cada configuración
        # LÓGICA: Agrupar por alpha y calcular promedios de todas las métricas
        alpha_analysis = {}
        for alpha in self.alpha_values:
            alpha_data = df.filter(pl.col("alpha") == alpha)
            if not alpha_data.is_empty():
                alpha_analysis[alpha] = {
                    "name": self._get_alpha_name(alpha),
                    # Métricas de precisión por posición (documento)
                    "document_hit_at_1_mean": alpha_data["document_hit_at_1"].mean(),
                    "document_hit_at_2_mean": alpha_data["document_hit_at_2"].mean(),
                    "document_hit_at_3_mean": alpha_data["document_hit_at_3"].mean(),
                    "document_hit_at_5_mean": alpha_data["document_hit_at_5"].mean(),
                    "document_hit_at_10_mean": alpha_data["document_hit_at_10"].mean(),
                    # Métricas de precisión por posición (chunk)
                    "chunk_hit_at_1_mean": alpha_data["chunk_hit_at_1"].mean(),
                    "chunk_hit_at_2_mean": alpha_data["chunk_hit_at_2"].mean(),
                    "chunk_hit_at_3_mean": alpha_data["chunk_hit_at_3"].mean(),
                    "chunk_hit_at_5_mean": alpha_data["chunk_hit_at_5"].mean(),
                    "chunk_hit_at_10_mean": alpha_data["chunk_hit_at_10"].mean(),
                    # Métricas de calidad del ranking
                    "document_mrr_mean": alpha_data["document_mrr"].mean(),
                    "chunk_mrr_mean": alpha_data["chunk_mrr"].mean(),
                    # Métricas de confianza y rendimiento
                    "best_score_mean": alpha_data["best_score"].mean(),
                    "f1_score_mean": alpha_data["f1_score"].mean() if self.generate_answers else 0,
                    "semantic_similarity_mean": alpha_data["semantic_similarity"].mean() if self.generate_answers else 0,
                    "total_time_mean": alpha_data["total_time"].mean(),
                }
        
        # Mejores configuraciones: Identificar el alpha óptimo para cada métrica
        # LÓGICA: Encontrar el alpha que maximiza cada métrica individual
        # IMPORTANCIA: Permite identificar configuraciones específicas para diferentes objetivos
        
        # Mejores configuraciones para métricas de documento
        best_doc_hit1 = max(alpha_analysis.keys(), 
                           key=lambda a: alpha_analysis[a]["document_hit_at_1_mean"])
        best_doc_hit2 = max(alpha_analysis.keys(), 
                           key=lambda a: alpha_analysis[a]["document_hit_at_2_mean"])
        best_doc_hit3 = max(alpha_analysis.keys(), 
                           key=lambda a: alpha_analysis[a]["document_hit_at_3_mean"])
        best_doc_hit5 = max(alpha_analysis.keys(), 
                           key=lambda a: alpha_analysis[a]["document_hit_at_5_mean"])
        best_doc_hit10 = max(alpha_analysis.keys(), 
                            key=lambda a: alpha_analysis[a]["document_hit_at_10_mean"])
        
        # Mejores configuraciones para métricas de chunk (si están disponibles)
        best_chunk_hit1 = max(alpha_analysis.keys(),
                             key=lambda a: alpha_analysis[a]["chunk_hit_at_1_mean"])
        best_chunk_hit2 = max(alpha_analysis.keys(),
                             key=lambda a: alpha_analysis[a]["chunk_hit_at_2_mean"])
        best_chunk_hit3 = max(alpha_analysis.keys(),
                             key=lambda a: alpha_analysis[a]["chunk_hit_at_3_mean"])
        best_chunk_hit5 = max(alpha_analysis.keys(),
                             key=lambda a: alpha_analysis[a]["chunk_hit_at_5_mean"])
        best_chunk_hit10 = max(alpha_analysis.keys(),
                              key=lambda a: alpha_analysis[a]["chunk_hit_at_10_mean"])
        
        # Mejor configuración para F1 Score (si está disponible)
        best_f1_score = None
        if self.generate_answers and any(alpha_analysis[a]["f1_score_mean"] > 0 for a in alpha_analysis.keys()):
            best_f1_score = max(alpha_analysis.keys(),
                               key=lambda a: alpha_analysis[a]["f1_score_mean"])
        
        analysis = {
            "alpha_analysis": alpha_analysis,
            "best_configurations": {
                "document_hit_at_1": best_doc_hit1,
                "document_hit_at_2": best_doc_hit2,
                "document_hit_at_3": best_doc_hit3,
                "document_hit_at_5": best_doc_hit5,
                "document_hit_at_10": best_doc_hit10,
                "chunk_hit_at_1": best_chunk_hit1,
                "chunk_hit_at_2": best_chunk_hit2,
                "chunk_hit_at_3": best_chunk_hit3,
                "chunk_hit_at_5": best_chunk_hit5,
                "chunk_hit_at_10": best_chunk_hit10,
                "f1_score": best_f1_score,
            },
            "overall_stats": {
                "total_questions": len(df),
                "total_configurations": len(self.alpha_values),
                "avg_retrieval_time": df["retrieval_time"].mean(),
                "avg_generation_time": df["generation_time"].mean() if self.generate_answers else 0,
                "generation_metrics_available": self.generate_answers,
            }
        }
        
        return analysis
    
    def _create_visualizations(self, results: List[Dict]):
        """Crear visualizaciones: Document Hits (líneas), F1/Semantic (líneas + boxplots)."""
        df = pd.DataFrame(results)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        # Figura 1: Consolidado de Document Hits por Alpha (LÍNEAS)
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calcular promedios por alpha para todas las métricas Hit
        doc_hit1_by_alpha = df.groupby('alpha')['document_hit_at_1'].mean()
        doc_hit2_by_alpha = df.groupby('alpha')['document_hit_at_2'].mean()
        doc_hit3_by_alpha = df.groupby('alpha')['document_hit_at_3'].mean()
        doc_hit5_by_alpha = df.groupby('alpha')['document_hit_at_5'].mean()
        doc_hit10_by_alpha = df.groupby('alpha')['document_hit_at_10'].mean()
        
        # Plot consolidado de todas las métricas Document Hit
        ax.plot(doc_hit1_by_alpha.index, doc_hit1_by_alpha.values,
               marker='o', linewidth=2, color='blue', label='Hit@1', markersize=4)
        ax.plot(doc_hit2_by_alpha.index, doc_hit2_by_alpha.values,
               marker='s', linewidth=2, color='green', label='Hit@2', markersize=4)
        ax.plot(doc_hit3_by_alpha.index, doc_hit3_by_alpha.values,
               marker='^', linewidth=2, color='red', label='Hit@3', markersize=4)
        ax.plot(doc_hit5_by_alpha.index, doc_hit5_by_alpha.values,
               marker='D', linewidth=2, color='purple', label='Hit@5', markersize=4)
        ax.plot(doc_hit10_by_alpha.index, doc_hit10_by_alpha.values,
               marker='*', linewidth=2, color='orange', label='Hit@10', markersize=4)
        
        # Añadir etiquetas de valores en cada punto
        for idx, val in zip(doc_hit1_by_alpha.index, doc_hit1_by_alpha.values):
            ax.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", 
                       xytext=(0,5), ha='center', fontsize=7, color='blue')
        for idx, val in zip(doc_hit2_by_alpha.index, doc_hit2_by_alpha.values):
            ax.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", 
                       xytext=(0,5), ha='center', fontsize=7, color='green')
        for idx, val in zip(doc_hit3_by_alpha.index, doc_hit3_by_alpha.values):
            ax.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", 
                       xytext=(0,5), ha='center', fontsize=7, color='red')
        for idx, val in zip(doc_hit5_by_alpha.index, doc_hit5_by_alpha.values):
            ax.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", 
                       xytext=(0,5), ha='center', fontsize=7, color='purple')
        for idx, val in zip(doc_hit10_by_alpha.index, doc_hit10_by_alpha.values):
            ax.annotate(f'{val:.2f}', (idx, val), textcoords="offset points", 
                       xytext=(0,5), ha='center', fontsize=7, color='orange')
        
        ax.set_title('Document Hits por Alpha (Saltos de 0.05)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Alpha', fontsize=12)
        ax.set_ylabel('Document Hit Score', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(np.arange(0, 1.05, 0.05))
        ax.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        plot_file = self.results_dir / "document_hits_by_alpha.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Gráfico Document Hits guardado: {plot_file}")
        
        # Figura 2: Box plot de MRR con media y mediana por alpha
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        
        # Preparar datos para box plot: Agrupar por alpha
        alpha_labels = [f"{a:.2f}" for a in sorted(df['alpha'].unique())]
        
        # MRR box plot
        mrr_data = [df[df['alpha'] == alpha]['document_mrr'].values
                   for alpha in sorted(df['alpha'].unique())]
        bp_mrr = ax.boxplot(mrr_data, labels=alpha_labels, patch_artist=True)
        
        # Customize box plot colors
        for patch in bp_mrr['boxes']:
            patch.set_facecolor('lightyellow')
            patch.set_alpha(0.7)
        
        # Calcular y mostrar media y mediana para cada alpha
        mrr_means = [np.mean(data) for data in mrr_data]
        mrr_medians = [np.median(data) for data in mrr_data]
        
        # Plot media y mediana como líneas
        ax.plot(range(1, len(mrr_means) + 1), mrr_means, 
                color='darkgoldenrod', linestyle='--', linewidth=2, marker='o', 
                markersize=3, label='Media', zorder=3)
        ax.plot(range(1, len(mrr_medians) + 1), mrr_medians, 
                color='black', linestyle='-', linewidth=2, marker='s', 
                markersize=3, label='Mediana', zorder=3)
        
        ax.set_title('Distribución MRR (Mean Reciprocal Rank) por Alpha', fontsize=12, fontweight='bold')
        ax.set_xlabel('Alpha', fontsize=10)
        ax.set_ylabel('MRR', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        mrr_file = self.results_dir / "mrr_boxplot_by_alpha.png"
        plt.savefig(mrr_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Box plot MRR guardado: {mrr_file}")
        
        # Figura 3: Box plots de F1 Score y Semantic Similarity con media y mediana (solo si generación habilitada)
        if self.generate_answers:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # F1 Score box plot
            f1_data = [df[df['alpha'] == alpha]['f1_score'].values
                      for alpha in sorted(df['alpha'].unique())]
            bp1 = axes[0].boxplot(f1_data, labels=alpha_labels, patch_artist=True)
            
            # Customize box plot colors
            for patch in bp1['boxes']:
                patch.set_facecolor('lightcoral')
                patch.set_alpha(0.7)
            
            # Calcular y mostrar media y mediana para cada alpha
            f1_means = [np.mean(data) for data in f1_data]
            f1_medians = [np.median(data) for data in f1_data]
            
            # Plot media y mediana como líneas
            axes[0].plot(range(1, len(f1_means) + 1), f1_means, 
                        color='darkred', linestyle='--', linewidth=2, marker='o', 
                        markersize=3, label='Media', zorder=3)
            axes[0].plot(range(1, len(f1_medians) + 1), f1_medians, 
                        color='black', linestyle='-', linewidth=2, marker='s', 
                        markersize=3, label='Mediana', zorder=3)
            
            axes[0].set_title('Distribución F1 Score por Alpha', fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Alpha', fontsize=10)
            axes[0].set_ylabel('F1 Score', fontsize=10)
            axes[0].grid(True, alpha=0.3, axis='y')
            axes[0].legend(fontsize=9)
            axes[0].tick_params(axis='x', rotation=45)
            
            # Semantic Similarity box plot
            semantic_data = [df[df['alpha'] == alpha]['semantic_similarity'].values
                           for alpha in sorted(df['alpha'].unique())]
            bp2 = axes[1].boxplot(semantic_data, labels=alpha_labels, patch_artist=True)
            
            # Customize box plot colors
            for patch in bp2['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
            
            # Calcular y mostrar media y mediana para cada alpha
            semantic_means = [np.mean(data) for data in semantic_data]
            semantic_medians = [np.median(data) for data in semantic_data]
            
            # Plot media y mediana como líneas
            axes[1].plot(range(1, len(semantic_means) + 1), semantic_means, 
                        color='darkblue', linestyle='--', linewidth=2, marker='o', 
                        markersize=3, label='Media', zorder=3)
            axes[1].plot(range(1, len(semantic_medians) + 1), semantic_medians, 
                        color='black', linestyle='-', linewidth=2, marker='s', 
                        markersize=3, label='Mediana', zorder=3)
            
            axes[1].set_title('Distribución Semantic Similarity por Alpha', fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Alpha', fontsize=10)
            axes[1].set_ylabel('Semantic Similarity', fontsize=10)
            axes[1].grid(True, alpha=0.3, axis='y')
            axes[1].legend(fontsize=9)
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            boxplot_file = self.results_dir / "generation_metrics_boxplots.png"
            plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Box plots F1/Semantic guardados: {boxplot_file}")
        else:
            print("ℹ️  Generación deshabilitada - No se generan gráficos de F1 y Semantic Similarity")
    
    def _print_final_summary(self, analysis: Dict):
        """Imprimir resumen final de la evaluación."""
        print(f"\n📋 RESUMEN FINAL DE EVALUACIÓN")
        print("=" * 80)
        
        overall = analysis["overall_stats"]
        best_configs = analysis["best_configurations"]
        
        print(f"📊 Estadísticas generales:")
        print(f"   • Total preguntas evaluadas: {overall['total_questions']}")
        print(f"   • Configuraciones alpha probadas: {overall['total_configurations']}")
        print(f"   • Tiempo promedio recuperación: {overall['avg_retrieval_time']:.2f}s")
        if overall['avg_generation_time'] > 0:
            print(f"   • Tiempo promedio generación: {overall['avg_generation_time']:.2f}s")
        
        print(f"\n🏆 Mejores configuraciones:")
        
        # Document-level (SIEMPRE disponible)
        # LÓGICA: Mostrar las mejores configuraciones para cada métrica de precisión
        best_doc_hit1_alpha = best_configs["document_hit_at_1"]
        best_doc_hit2_alpha = best_configs["document_hit_at_2"]
        best_doc_hit3_alpha = best_configs["document_hit_at_3"]
        best_doc_hit5_alpha = best_configs["document_hit_at_5"]
        best_doc_hit10_alpha = best_configs["document_hit_at_10"]
        print(f"   🔍 DOCUMENT-LEVEL:")
        print(f"     • Mejor Document Hit@1: α={best_doc_hit1_alpha:.2f} "
              f"({analysis['alpha_analysis'][best_doc_hit1_alpha]['name']})")
        print(f"       Score: {analysis['alpha_analysis'][best_doc_hit1_alpha]['document_hit_at_1_mean']:.3f}")
        print(f"     • Mejor Document Hit@2: α={best_doc_hit2_alpha:.2f} "
              f"({analysis['alpha_analysis'][best_doc_hit2_alpha]['name']})")
        print(f"       Score: {analysis['alpha_analysis'][best_doc_hit2_alpha]['document_hit_at_2_mean']:.3f}")
        print(f"     • Mejor Document Hit@3: α={best_doc_hit3_alpha:.2f} "
              f"({analysis['alpha_analysis'][best_doc_hit3_alpha]['name']})")
        print(f"       Score: {analysis['alpha_analysis'][best_doc_hit3_alpha]['document_hit_at_3_mean']:.3f}")
        print(f"     • Mejor Document Hit@5: α={best_doc_hit5_alpha:.2f} "
              f"({analysis['alpha_analysis'][best_doc_hit5_alpha]['name']})")
        print(f"       Score: {analysis['alpha_analysis'][best_doc_hit5_alpha]['document_hit_at_5_mean']:.3f}")
        print(f"     • Mejor Document Hit@10: α={best_doc_hit10_alpha:.2f} "
              f"({analysis['alpha_analysis'][best_doc_hit10_alpha]['name']})")
        print(f"       Score: {analysis['alpha_analysis'][best_doc_hit10_alpha]['document_hit_at_10_mean']:.3f}")
        
        # Chunk-level (SOLO si disponible)
        # LÓGICA: Solo mostrar si hay ground truth a nivel de chunk disponible
        if self.has_chunk_ids:
            best_chunk_hit1_alpha = best_configs["chunk_hit_at_1"]
            best_chunk_hit2_alpha = best_configs["chunk_hit_at_2"]
            best_chunk_hit3_alpha = best_configs["chunk_hit_at_3"]
            best_chunk_hit5_alpha = best_configs["chunk_hit_at_5"]
            best_chunk_hit10_alpha = best_configs["chunk_hit_at_10"]
            print(f"   📄 CHUNK-LEVEL:")
            print(f"     • Mejor Chunk Hit@1: α={best_chunk_hit1_alpha:.2f} "
                  f"({analysis['alpha_analysis'][best_chunk_hit1_alpha]['name']})")
            print(f"       Score: {analysis['alpha_analysis'][best_chunk_hit1_alpha]['chunk_hit_at_1_mean']:.3f}")
            print(f"     • Mejor Chunk Hit@2: α={best_chunk_hit2_alpha:.2f} "
                  f"({analysis['alpha_analysis'][best_chunk_hit2_alpha]['name']})")
            print(f"       Score: {analysis['alpha_analysis'][best_chunk_hit2_alpha]['chunk_hit_at_2_mean']:.3f}")
            print(f"     • Mejor Chunk Hit@3: α={best_chunk_hit3_alpha:.2f} "
                  f"({analysis['alpha_analysis'][best_chunk_hit3_alpha]['name']})")
            print(f"       Score: {analysis['alpha_analysis'][best_chunk_hit3_alpha]['chunk_hit_at_3_mean']:.3f}")
            print(f"     • Mejor Chunk Hit@5: α={best_chunk_hit5_alpha:.2f} "
                  f"({analysis['alpha_analysis'][best_chunk_hit5_alpha]['name']})")
            print(f"       Score: {analysis['alpha_analysis'][best_chunk_hit5_alpha]['chunk_hit_at_5_mean']:.3f}")
            print(f"     • Mejor Chunk Hit@10: α={best_chunk_hit10_alpha:.2f} "
                  f"({analysis['alpha_analysis'][best_chunk_hit10_alpha]['name']})")
            print(f"       Score: {analysis['alpha_analysis'][best_chunk_hit10_alpha]['chunk_hit_at_10_mean']:.3f}")
        else:
            print(f"   📄 CHUNK-LEVEL: No disponible (sin gold_chunk_ids en dataset)")
        
        # Generation-level (SOLO si está habilitado) - F1 Score y Semantic Similarity
        if self.generate_answers and best_configs["f1_score"]:
            best_f1_alpha = best_configs["f1_score"]
            print(f"   🎯 GENERATION-LEVEL:")
            print(f"     • Mejor F1 Score: α={best_f1_alpha:.2f} "
                  f"({analysis['alpha_analysis'][best_f1_alpha]['name']})")
            print(f"       Score: {analysis['alpha_analysis'][best_f1_alpha]['f1_score_mean']:.3f}")
        elif self.generate_answers:
            print(f"   🎯 GENERATION-LEVEL: Habilitado pero sin scores significativos")
        else:
            print(f"   🎯 GENERATION-LEVEL: No habilitado (generate_answers={self.generate_answers})")
        
        print(f"\n💡 Recomendaciones:")
        
        # Analizar patrones: Identificar la mejor configuración general
        # LÓGICA: Usar Hit@1 como métrica principal para determinar la mejor configuración
        # IMPORTANCIA: Hit@1 es la métrica más estricta y representa la precisión en la primera posición
        alpha_analysis = analysis["alpha_analysis"]
        doc_scores = [(a, data["document_hit_at_1_mean"]) for a, data in alpha_analysis.items()]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        top_alpha = doc_scores[0][0]
        
        if top_alpha == 0.0:
            print("   ✅ La búsqueda por palabras clave (BM25) funciona mejor")
            print("   ✅ Recomendado: usar α=0.0 o valores bajos de alpha")
        elif top_alpha == 1.0:
            print("   ✅ La búsqueda semántica funciona mejor")
            print("   ✅ Recomendado: usar α=1.0 o valores altos de alpha")
        else:
            print("   ✅ El enfoque híbrido funciona mejor")
            print(f"   ✅ Recomendado: usar α={top_alpha:.1f} como configuración principal")
        
        print(f"\n📁 Archivo principal generado:")
        print("   • evaluation_results/rag_evaluation_results.csv - Resultados consolidados (modo APPEND)")
        print("   ℹ️  Los resultados se agregan al archivo existente en cada ejecución")
        print("\n   📊 GRÁFICOS: Comentados por defecto para optimizar ejecución")
        print("       Para habilitar gráficos, descomentar línea en run_full_evaluation()")
        # print("   • retrieval_metrics_by_alpha.png - Gráficos de métricas de retrieval")
        # print("   • distribution_boxplots.png - Distribuciones de retrieval")
        # if self.generate_answers:
        #     print("   • generation_metrics_by_alpha.png - Métricas de generación (EXACT MATCH)")
        #     print("   • generation_distribution_boxplots.png - Distribuciones de generación")
        
        print(f"\n🎓 CUMPLIMIENTO DE RECOMENDACIONES DEL EXPERTO:")
        print("   ✅ Evaluación dinámica según disponibilidad de gold_chunk_ids")
        if self.has_chunk_ids:
            print("   ✅ Modo B: document@k + chunk@k + MRR + nDCG (granularidad fina)")
            print("   ✅ Diagnóstico preciso de recuperación a nivel chunk")
        else:
            print("   ✅ Modo A: solo document@k + MRR + nDCG (sin chunk ground truth)")
            print("   ✅ Métricas apropiadas para la información disponible")
        print("   ✅ Sin reportar métricas chunk-level cuando no hay gold_chunk_ids")
        print("   ✅ Separación clara entre document-level y chunk-level evaluation")
        if self.generate_answers:
            print("   ✅ Métricas de generación validadas por expertos implementadas:")
            print("      🎯 EXACT MATCH - Métrica clave para evaluación determinista")
            print("      📊 F1 Score - Token-level overlap para paráfrasis")
            print("      🧠 Similitud Semántica - Embedding-based similarity")
            print("   ✅ MEJORAS PARA EXACT MATCH IMPLEMENTADAS:")
            print("      🔧 Modo evaluación con salida estricta (sin cortesías)")
            print("      🔧 Normalización ampliada (números, espacios, porcentajes)")
            print("      🔧 Extracción directa del contexto para scores altos (≥0.8)")
            print("      🔧 Formato JSON opcional para máxima precisión")
            print("      🔧 Debug de comparaciones EM=0 para análisis")
        else:
            print("   ✅ Evaluación optimizada para retrieval (sin generación costosa)")


def main():
    """Función principal."""
    import argparse
    
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description="Sistema de evaluación RAG simplificado compatible con qlora_pref_train.py"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="llama3.2",
        help="Identificador del modelo para Ollama (detecta automáticamente tipo: base/dpo/orpo)"
    )
    parser.add_argument(
        "--embedder_id",
        type=str,
        default="nomic-embed-text-v2",
        help="Modelo de embeddings"
    )
    parser.add_argument(
        "--golden_set",
        type=str,
        default="golden_set_test.csv",
        help="Archivo CSV del conjunto golden"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="evaluation_results",
        help="Directorio para guardar resultados"
    )
    parser.add_argument(
        "--generate_answers",
        action="store_true",
        default=True,
        help="Generar respuestas con el modelo (incluye métricas F1)"
    )
    parser.add_argument(
        "--no_generate_answers",
        action="store_true",
        help="No generar respuestas (solo evaluación de retrieval)"
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="System prompt para el modelo. Por defecto vacío para evaluación justa (sin sesgos)"
    )
    parser.add_argument(
        "--rag_mode",
        type=str,
        choices=["hybrid", "advanced", "both"],
        default="both",
        help="Modo de RAG: 'hybrid', 'advanced', o 'both' (ejecuta ambos)"
    )
    parser.add_argument(
        "--translation",
        type=str,
        choices=["enabled", "disabled", "both"],
        default="both",
        help="Traducción: 'enabled', 'disabled', o 'both' (ejecuta ambos)"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Seleccionar GPU (default: 0)"
    )
    
    args = parser.parse_args()
    
    # Configurar GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # Resolver conflicto de flags
    generate_answers = args.generate_answers and not args.no_generate_answers
    
    print("🔍 SISTEMA DE EVALUACIÓN RAG SIMPLIFICADO")
    print("=" * 80)
    print(f"📋 Parámetros:")
    print(f"   • Modelo: {args.model_id}")
    print(f"   • Embedder: {args.embedder_id}")
    print(f"   • Golden set: {args.golden_set}")
    print(f"   • Generar respuestas: {generate_answers}")
    print(f"   • Modo RAG: {args.rag_mode}")
    print(f"   • Traducción: {args.translation}")
    print(f"   • GPU: {args.gpu}")
    print(f"   • System prompt: {'(vacío - evaluación justa)' if args.system_prompt == '' else f'({len(args.system_prompt)} chars)'}")
    print()
    
    # Verificar archivos necesarios
    if not Path(args.golden_set).exists():
        print(f"❌ Archivo {args.golden_set} no encontrado")
        return 1
    
    if not Path("tmp/lancedb").exists():
        print("❌ Base de datos LanceDB no encontrada en tmp/lancedb")
        print("   Ejecuta 'python generate-qhaliknowdb.py' primero")
        return 1
    
    # Determinar combinaciones de evaluación a ejecutar
    rag_modes = []
    if args.rag_mode == "both":
        rag_modes = ["hybrid", "advanced"]
    else:
        rag_modes = [args.rag_mode]
    
    translation_modes = []
    if args.translation == "both":
        translation_modes = [True, False]
    else:
        translation_modes = [args.translation == "enabled"]
    
    print(f"\n🔄 Se ejecutarán {len(rag_modes) * len(translation_modes)} configuraciones:")
    for rag_mode in rag_modes:
        for use_translation in translation_modes:
            print(f"   • RAG: {rag_mode}, Traducción: {'✅' if use_translation else '❌'}")
    print()
    
    # Ejecutar evaluación para cada combinación
    all_configs_results = []
    
    try:
        for rag_mode in rag_modes:
            for use_translation in translation_modes:
                config_name = f"{rag_mode}_translation_{use_translation}"
                print(f"\n{'='*80}")
                print(f"🚀 CONFIGURACIÓN: RAG={rag_mode.upper()}, Traducción={'HABILITADA' if use_translation else 'DESHABILITADA'}")
                print(f"{'='*80}\n")
                
                # Configuración del agente
                print("🤖 Inicializando RAGAgent...")
                
                # Parámetros extra para modo advanced (alineados con test_rag_agent)
                advanced_kwargs = {}
                if rag_mode == "advanced":
                    advanced_kwargs = dict(
                        adv_num_queries=3,
                        adv_top_k_per_query=5,
                        adv_merge_strategy="vote+score",
                        adv_rerank_strategy="mmr",
                        adv_max_chunks=6,
                    )
                    print("⚙️  Parámetros Advanced configurados:")
                    print(f"   • adv_num_queries: 3")
                    print(f"   • adv_top_k_per_query: 5")
                    print(f"   • adv_max_chunks: 6")
                    print(f"   • merge_strategy: vote+score")
                    print(f"   • rerank_strategy: mmr")
                
                try:
                    agent = RAGAgent(
                        model_id=args.model_id,
                        embedder_id=args.embedder_id,
                        lancedb_path="tmp/lancedb",
                        table_name="docs_qa",
                        top_k_semantic=50,  # Usar más documentos para evaluación
                        top_k_keyword=50,
                        alpha=0.7,  # Default, será sobrescrito en evaluación
                        similarity_threshold=0.0,  # Sin filtro para evaluación
                        auto_warm_up=False,
                        enable_translation=use_translation,
                        silent_translation=True,  # Silenciar logs de traducción
                        skip_chunk_translation=False,  # Mantener traducción de chunks
                        system_prompt=args.system_prompt,  # System prompt configurable
                        mode=rag_mode,  # Establecer el modo de RAG (parámetro correcto)
                        **advanced_kwargs,
                    )
                    print("✅ RAGAgent inicializado correctamente")
                    if args.system_prompt == "":
                        print("   ⚠️  System prompt vacío - Evaluación justa sin sesgos")
                    
                except Exception as e:
                    print(f"❌ Error inicializando RAGAgent: {e}")
                    continue  # Continuar con la siguiente configuración
                
                # Configurar sistema de evaluación con modo específico
                results_subdir = f"{args.results_dir}_{rag_mode}_trans_{use_translation}"
                evaluation_system = SimpleRAGEvaluationSystem(
                    agent=agent,
                    golden_set_file=args.golden_set,
                    results_dir=results_subdir,
                    generate_answers=generate_answers,
                    verbose=True,
                    model_id=args.model_id,
                    rag_mode=rag_mode,
                    use_translation=use_translation
                )
                
                # Ejecutar evaluación completa
                try:
                    results = evaluation_system.run_full_evaluation()
                    all_configs_results.append({
                        "config_name": config_name,
                        "rag_mode": rag_mode,
                        "use_translation": use_translation,
                        "results": results
                    })
                    print(f"\n✅ EVALUACIÓN COMPLETADA: {config_name}")
                except Exception as e:
                    print(f"\n❌ Error en evaluación {config_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Resumen final de todas las configuraciones
        if len(all_configs_results) > 1:
            print(f"\n{'='*80}")
            print(f"📊 RESUMEN DE TODAS LAS CONFIGURACIONES")
            print(f"{'='*80}")
            print(f"\nSe completaron {len(all_configs_results)} de {len(rag_modes) * len(translation_modes)} configuraciones:")
            for config_result in all_configs_results:
                print(f"   ✅ {config_result['config_name']}")
        
        if all_configs_results:
            print("\n✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
            return 0
        else:
            print("\n❌ No se completó ninguna evaluación")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Evaluación interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"\n❌ Error durante la evaluación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
