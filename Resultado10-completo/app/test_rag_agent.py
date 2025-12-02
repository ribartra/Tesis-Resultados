#!/usr/bin/env python
"""
Script de pruebas unitarias para RAGAgent
Evalúa el pipeline RAG en modo hybrid y advanced usando golden_set_test.csv
"""

import pytest
import pandas as pd
import json
import time
import sys
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from rag_agent import RAGAgent


class DualOutput:
    """Clase para escribir simultáneamente a stdout y archivo"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class TestRAGAgent:
    """Suite de pruebas para RAGAgent en modos hybrid y advanced"""
    
    # Variable de clase para almacenar el archivo de resultados
    results_file = None
    all_logs = []
    
    @classmethod
    def setup_class(cls):
        """Configura el archivo de salida antes de ejecutar los tests"""
        # Abrir archivo en modo escritura
        cls.results_file = open("test_rag_agent_results.txt", "w", encoding="utf-8")
        cls.all_logs = []
        
        # Escribir encabezado
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = [
            "=" * 80,
            "TEST RAG AGENT - Resultados de Ejecución",
            f"Fecha: {timestamp}",
            "=" * 80,
            ""
        ]
        for line in header:
            print(line)
            cls.all_logs.append(line)
    
    @classmethod
    def teardown_class(cls):
        """Guarda todos los logs y cierra el archivo de salida"""
        if cls.results_file:
            # Escribir todos los logs acumulados
            cls.results_file.write("\n".join(cls.all_logs))
            
            # Escribir footer
            footer = [
                "",
                "=" * 80,
                "FIN DE LA EJECUCIÓN",
                "Resultados guardados en: test_rag_agent_results.txt",
                "=" * 80
            ]
            for line in footer:
                print(line)
                cls.results_file.write(line + "\n")
            
            cls.results_file.close()
            print(f"\n✓ Archivo 'test_rag_agent_results.txt' guardado exitosamente\n")
    
    @pytest.fixture(scope="class")
    def golden_data(self):
        """Carga las primeras 10 filas del golden set"""
        df = pd.read_csv("golden_set_test.csv")
        return df.head(10)
    
    @pytest.fixture(scope="class")
    def docs_metadata(self):
        """Carga metadatos de documentos para validación"""
        with open("docs_metadata.json", "r") as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def available_docs(self, docs_metadata):
        """Extrae lista de documentos disponibles en la base de conocimiento"""
        return list(docs_metadata.get("documents", {}).keys())
    
    def log_execution(self, message: str, level: str = "INFO"):
        """Helper para logging con timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] [{level}] {message}"
        print(log_line)
        # Guardar en la lista de logs para escribir al final
        TestRAGAgent.all_logs.append(log_line)
    
    def extract_document_names(self, gold_doc_ids: str) -> List[str]:
        """Extrae nombres de documentos del campo gold_doc_ids"""
        if pd.isna(gold_doc_ids):
            return []
        # Puede venir separado por comas
        return [doc.strip() for doc in str(gold_doc_ids).split(",")]
    
    def validate_context_exists(
        self, 
        retrieved_chunks: List[Dict], 
        gold_docs: List[str],
        available_docs: List[str]
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Valida si el contexto recuperado contiene documentos del golden set
        
        Returns:
            (found, retrieved_doc_names, matched_docs)
        """
        # Extraer nombres únicos de documentos recuperados
        retrieved_doc_names = list(set([
            chunk.get("pdf_name", "") for chunk in retrieved_chunks
        ]))
        
        # Verificar cuáles documentos gold están en los recuperados
        matched_docs = []
        for gold_doc in gold_docs:
            # Verificar si el documento existe en la base de conocimiento
            if gold_doc not in available_docs:
                self.log_execution(
                    f"  ⚠️  Documento gold '{gold_doc}' NO está en la base de conocimiento",
                    "WARNING"
                )
                continue
            
            # Verificar si fue recuperado
            if gold_doc in retrieved_doc_names:
                matched_docs.append(gold_doc)
        
        # Se considera que el contexto existe si se recuperó al menos 1 documento gold
        context_found = len(matched_docs) > 0
        
        return context_found, retrieved_doc_names, matched_docs
    
    def log_retrieval_results(
        self,
        qid: int,
        question: str,
        retrieved_chunks: List[Dict],
        gold_docs: List[str],
        matched_docs: List[str],
        context_found: bool,
        execution_time: float
    ):
        """Imprime log detallado de los resultados de recuperación"""
        self.log_execution("=" * 80)
        self.log_execution(f"QID {qid}: {question}")
        self.log_execution("-" * 80)
        self.log_execution(f"Documentos esperados (gold): {', '.join(gold_docs) if gold_docs else 'N/A'}")
        self.log_execution(f"Total de chunks recuperados: {len(retrieved_chunks)}")
        
        if retrieved_chunks:
            self.log_execution("Documentos recuperados y citados:")
            
            # Agrupar chunks por documento
            docs_dict = {}
            for chunk in retrieved_chunks:
                doc_name = chunk.get("pdf_name", "unknown")
                if doc_name not in docs_dict:
                    docs_dict[doc_name] = []
                docs_dict[doc_name].append(chunk)
            
            # Imprimir cada documento con sus chunks
            for doc_name, chunks in docs_dict.items():
                avg_score = sum(c.get("score_final", 0) for c in chunks) / len(chunks)
                is_gold = "✓ GOLD" if doc_name in gold_docs else ""
                self.log_execution(f"  📄 {doc_name} {is_gold}")
                self.log_execution(f"     - Chunks: {len(chunks)}")
                self.log_execution(f"     - Score híbrido promedio: {avg_score:.4f}")
                
                # Mostrar top 3 chunks con mejor score
                sorted_chunks = sorted(chunks, key=lambda x: x.get("score_final", 0), reverse=True)
                for i, chunk in enumerate(sorted_chunks[:3], 1):
                    score = chunk.get("score_final", 0)
                    chunk_idx = chunk.get("chunk_index", "?")
                    self.log_execution(f"       {i}. Chunk #{chunk_idx} - Score: {score:.4f}")
        else:
            self.log_execution("  ⚠️  No se recuperaron chunks")
        
        self.log_execution("-" * 80)
        self.log_execution(f"Documentos gold recuperados: {', '.join(matched_docs) if matched_docs else 'Ninguno'}")
        self.log_execution(f"Contexto encontrado: {'✓ SÍ' if context_found else '✗ NO'}")
        self.log_execution(f"Tiempo de ejecución: {execution_time:.3f}s")
        self.log_execution("=" * 80)
        self.log_execution("")
    
    def run_rag_evaluation(
        self,
        agent: RAGAgent,
        golden_data: pd.DataFrame,
        available_docs: List[str],
        mode: str,
        alpha: float
    ) -> Tuple[int, int, List[Dict]]:
        """
        Ejecuta evaluación completa del RAG
        
        Returns:
            (total_queries, contexts_found, detailed_results)
        """
        self.log_execution("")
        self.log_execution("=" * 80)
        self.log_execution(f"INICIANDO EVALUACIÓN - MODO: {mode.upper()} | ALPHA: {alpha}")
        self.log_execution("=" * 80)
        self.log_execution("")
        
        total_queries = len(golden_data)
        contexts_found = 0
        detailed_results = []
        
        for idx, row in golden_data.iterrows():
            qid = row["qid"]
            question = row["question"]
            gold_doc_ids = row.get("gold_doc_ids", "")
            
            # Extraer documentos esperados
            gold_docs = self.extract_document_names(gold_doc_ids)
            
            # Medir tiempo de ejecución
            start_time = time.time()
            
            # Ejecutar recuperación con el agente
            try:
                retrieved_chunks = agent.retrieve_only(
                    query=question,
                    alpha=alpha
                )
            except Exception as e:
                self.log_execution(f"❌ Error en recuperación QID {qid}: {e}", "ERROR")
                retrieved_chunks = []
            
            execution_time = time.time() - start_time
            
            # Validar contexto
            context_found, retrieved_doc_names, matched_docs = self.validate_context_exists(
                retrieved_chunks, gold_docs, available_docs
            )
            
            if context_found:
                contexts_found += 1
            
            # Log detallado
            self.log_retrieval_results(
                qid, question, retrieved_chunks, gold_docs, 
                matched_docs, context_found, execution_time
            )
            
            # Guardar resultados
            detailed_results.append({
                "qid": qid,
                "question": question,
                "gold_docs": gold_docs,
                "retrieved_docs": retrieved_doc_names,
                "matched_docs": matched_docs,
                "context_found": context_found,
                "num_chunks": len(retrieved_chunks),
                "execution_time": execution_time,
                "mode": mode,
                "alpha": alpha
            })
        
        return total_queries, contexts_found, detailed_results
    
    def test_rag_hybrid_mode(self, golden_data, available_docs):
        """
        Test del RAG en modo HYBRID con alpha=0.65
        Debe encontrar contexto en al menos 80% de las consultas
        """
        self.log_execution("\n" + "🔵" * 40)
        self.log_execution("TEST: RAG HYBRID MODE")
        self.log_execution("🔵" * 40 + "\n")
        
        # Configurar agente en modo hybrid con warm-up habilitado
        self.log_execution("Inicializando agente con warm-up...")
        agent = RAGAgent(
            mode="hybrid",
            alpha=0.65,
            auto_warm_up=True,  # Habilitado para mejorar tiempos de respuesta
            enable_translation=True,
            silent_translation=True,
            skip_chunk_translation=False
        )
        
        self.log_execution(f"✓ Agente configurado: mode={agent.mode}, alpha={agent.alpha}")
        self.log_execution(f"✓ Traducción habilitada: {agent.enable_translation}")
        self.log_execution(f"✓ Warm-up completado")
        
        # Ejecutar evaluación
        total, found, results = self.run_rag_evaluation(
            agent, golden_data, available_docs, "hybrid", 0.65
        )
        
        # Calcular porcentaje
        success_rate = (found / total) * 100 if total > 0 else 0
        
        # Log resumen
        self.log_execution("")
        self.log_execution("=" * 80)
        self.log_execution("RESUMEN - MODO HYBRID")
        self.log_execution("=" * 80)
        self.log_execution(f"Total de consultas: {total}")
        self.log_execution(f"Contextos encontrados: {found}")
        self.log_execution(f"Tasa de éxito: {success_rate:.2f}%")
        self.log_execution(f"Umbral requerido: 80.00%")
        self.log_execution("=" * 80)
        self.log_execution("")
        
        # Validar que se cumple el umbral del 80%
        assert success_rate >= 80.0, (
            f"FALLO: Tasa de éxito {success_rate:.2f}% < 80%. "
            f"Solo {found}/{total} consultas tuvieron contexto en la base de conocimiento."
        )
        
        self.log_execution("✅ TEST PASADO - MODO HYBRID")
    
    def test_rag_advanced_mode(self, golden_data, available_docs):
        """
        Test del RAG en modo ADVANCED con alpha=0.65
        Debe encontrar contexto en al menos 80% de las consultas
        """
        self.log_execution("\n" + "🟢" * 40)
        self.log_execution("TEST: RAG ADVANCED MODE")
        self.log_execution("🟢" * 40 + "\n")
        
        # Configurar agente en modo advanced con warm-up habilitado
        self.log_execution("Inicializando agente con warm-up...")
        agent = RAGAgent(
            mode="advanced",
            alpha=1.0,  # Semántico puro según indicación del usuario
            auto_warm_up=True,  # Habilitado para mejorar tiempos de respuesta
            enable_translation=True,
            silent_translation=True,
            skip_chunk_translation=False,
            adv_num_queries=3,
            adv_top_k_per_query=5,
            adv_merge_strategy="vote+score",
            adv_rerank_strategy="mmr",
            adv_max_chunks=6
        )
        
        self.log_execution(f"✓ Agente configurado: mode={agent.mode}, alpha={agent.alpha}")
        self.log_execution(f"✓ Traducción habilitada: {agent.enable_translation}")
        self.log_execution(f"✓ Parámetros Advanced: num_queries={agent.adv_num_queries}, "
                          f"max_chunks={agent.adv_max_chunks}, merge={agent.adv_merge_strategy}")
        self.log_execution(f"✓ Warm-up completado")
        
        # Ejecutar evaluación
        total, found, results = self.run_rag_evaluation(
            agent, golden_data, available_docs, "advanced", 0.65
        )
        
        # Calcular porcentaje
        success_rate = (found / total) * 100 if total > 0 else 0
        
        # Log resumen
        self.log_execution("")
        self.log_execution("=" * 80)
        self.log_execution("RESUMEN - MODO ADVANCED")
        self.log_execution("=" * 80)
        self.log_execution(f"Total de consultas: {total}")
        self.log_execution(f"Contextos encontrados: {found}")
        self.log_execution(f"Tasa de éxito: {success_rate:.2f}%")
        self.log_execution(f"Umbral requerido: 80.00%")
        self.log_execution("=" * 80)
        self.log_execution("")
        
        # Validar que se cumple el umbral del 80%
        assert success_rate >= 80.0, (
            f"FALLO: Tasa de éxito {success_rate:.2f}% < 80%. "
            f"Solo {found}/{total} consultas tuvieron contexto en la base de conocimiento."
        )
        
        self.log_execution("✅ TEST PASADO - MODO ADVANCED")


if __name__ == "__main__":
    """Permite ejecutar directamente sin pytest para debugging"""
    
    # Configurar salida dual (terminal + archivo)
    output_handler = DualOutput("test_rag_agent_results.txt")
    sys.stdout = output_handler
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 80)
    print(f"EJECUCIÓN DIRECTA - TEST RAG AGENT")
    print(f"Fecha: {timestamp}")
    print("=" * 80)
    print()
    
    # Instanciar clase de test
    test_suite = TestRAGAgent()
    
    # Cargar datos
    print("Cargando datos de prueba...")
    golden_data = pd.read_csv("golden_set_test.csv").head(10)
    
    with open("docs_metadata.json", "r") as f:
        docs_metadata = json.load(f)
    
    available_docs = list(docs_metadata.get("documents", {}).keys())
    
    print(f"✓ Cargadas {len(golden_data)} consultas del golden set")
    print(f"✓ Disponibles {len(available_docs)} documentos en la base de conocimiento\n")
    
    # Ejecutar tests
    try:
        print("\n🔵 Ejecutando test modo HYBRID...")
        test_suite.test_rag_hybrid_mode(golden_data, available_docs)
        
        print("\n🟢 Ejecutando test modo ADVANCED...")
        test_suite.test_rag_advanced_mode(golden_data, available_docs)
        
        print("\n" + "=" * 80)
        print("✅ TODOS LOS TESTS PASARON EXITOSAMENTE")
        print("=" * 80)
        print()
        print("Resultados guardados en: test_rag_agent_results.txt")
        print("=" * 80)
        print()
        
        output_handler.close()
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n❌ TEST FALLIDO: {e}\n")
        output_handler.close()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}\n")
        import traceback
        traceback.print_exc()
        output_handler.close()
        sys.exit(1)
