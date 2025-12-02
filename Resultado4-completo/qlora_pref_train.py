#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entrenamiento QLoRA (Unsloth) con DPO/ORPO sobre dataset CSV (Polars)
- Lectura y preprocesamiento con Polars de (prompt, chosen, rejected)
- Split determinista train/val/test por hash del prompt normalizado (evita fuga)
- Presets de hparams (lista) pero sólo se usa la posición --preset_index (default 0)
- Callback imprime al final de cada epoch: pérdida media, reducción vs epoch previa
  y nota sobre la función de pérdida (DPO u ORPO)
- --model_id (base) y --output_model_id (opcional). Si no se pasa, se autosetea con timestamp.
"""

from __future__ import annotations
import os
import re
import argparse
import hashlib
import sys
import subprocess
import time
import glob

from datetime import datetime
from typing import Dict, List, Tuple

import polars as pl
from datasets import Dataset
from collections import defaultdict

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import DPOTrainer, ORPOTrainer
from transformers import TrainingArguments, TrainerCallback

from pathlib import Path


import random
import numpy as np
try:
    import torch
except Exception:
    torch = None

# ---------------------------------
# Presets (escogemos solo uno)
# ---------------------------------
PRESETS = [
    # Posición 0 (recomendada, estable y eficiente)
    dict(
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
        lr_dpo=2e-6, lr_orpo=3e-6, beta=0.1,               # beta para DPO; ORPO usa "beta/lambda" ~ 0.1
        per_device_bs=2, grad_accum=8, warmup_ratio=0.1,   # Tamanio Batch efectivo≈16
        max_seq_len=1536, max_prompt_len=768,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    ),
    # Posición 1 (más capacidad)
    dict(
        lora_r=32, lora_alpha=64, lora_dropout=0.0,
        lr_dpo=3e-6, lr_orpo=5e-6, beta=0.1,
        per_device_bs=2, grad_accum=8, warmup_ratio=0.1,  # EBS≈16
        max_seq_len=1536, max_prompt_len=768,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    ),
    # Posición 2 (más conservador)
    dict(
        lora_r=8, lora_alpha=16, lora_dropout=0.1,
        lr_dpo=1e-6, lr_orpo=2e-6, beta=0.1,
        per_device_bs=2, grad_accum=8, warmup_ratio=0.1,
        max_seq_len=1536, max_prompt_len=768,
        target_modules=["q_proj","v_proj","o_proj","up_proj","down_proj"],
    ),
]

# ---------------------------------
# Utilidades de datos (Polars)
# ---------------------------------
def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def read_and_preprocess(csv_path: str) -> pl.DataFrame:
    df = pl.read_csv(csv_path, infer_schema_length=10000)
    needed = {"prompt","chosen","rejected"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"CSV debe contener columnas {needed}, contiene: {df.columns}")

    # Trim + eliminar nulos/vacíos + chosen!=rejected
    df = (
        df
        .with_columns([
            pl.col("prompt").cast(pl.Utf8).map_elements(_norm_text, return_dtype=pl.Utf8),
            pl.col("chosen").cast(pl.Utf8).map_elements(_norm_text, return_dtype=pl.Utf8),
            pl.col("rejected").cast(pl.Utf8).map_elements(_norm_text, return_dtype=pl.Utf8),
        ])
        .filter(
            (pl.col("prompt").str.len_chars() > 0) &
            (pl.col("chosen").str.len_chars() > 0) &
            (pl.col("rejected").str.len_chars() > 0) &
            (pl.col("chosen") != pl.col("rejected"))
        )
        .unique(subset=["prompt","chosen","rejected"])
    )

    # (Opcional) filtros de longitud para estabilidad de truncado
    df = df.filter(
        (pl.col("prompt").str.len_chars() <= 1200) &
        (pl.col("chosen").str.len_chars() <= 1200) &
        (pl.col("rejected").str.len_chars() <= 1200)
    )
    return df

def _hash_bucket(s: str) -> int:
    # bucket determinista 0..99 por prompt normalizado
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100

def split_df_hash(df: pl.DataFrame, train=0.8, val=0.1, test=0.1) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if abs((train+val+test) - 1.0) > 1e-6:
        raise ValueError("Ratios deben sumar 1.0")
    thresholds = (int(train*100), int((train+val)*100))
    df = df.with_columns(pl.col("prompt").map_elements(_norm_text, return_dtype=pl.Utf8).alias("_p_norm"))
    df = df.with_columns(pl.col("_p_norm").map_elements(_hash_bucket, return_dtype=pl.Int32).alias("_bucket"))

    train_df = df.filter(pl.col("_bucket") < thresholds[0]).drop(["_p_norm","_bucket"])
    val_df   = df.filter((pl.col("_bucket") >= thresholds[0]) & (pl.col("_bucket") < thresholds[1])).drop(["_p_norm","_bucket"])
    test_df  = df.filter(pl.col("_bucket") >= thresholds[1]).drop(["_p_norm","_bucket"])
    return train_df, val_df, test_df

def to_hf_dataset(df: pl.DataFrame) -> Dataset:
    data = {c: df.get_column(c).to_list() for c in ["prompt","chosen","rejected"]}
    return Dataset.from_dict(data)
    
    
# ---------------------------------
# Chat templates y limpieza de tokens (Llama 2 / 3.x, Qwen 2.5_3, Gemma 2_3)
# ---------------------------------
def detect_model_family(model_id: str) -> str:
    mid = (model_id or "").lower()
    # Llama 3.x
    if re.search(r"llama[-_]?3(\.2|\.1)?", mid):
        return "llama3"
    # Llama 2
    if "llama-2" in mid or "llama2" in mid:
        return "llama2"
    # Qwen 2.5 / 3
    if re.search(r"qwen[-_]?2\.5", mid):
        return "qwen2.5"
    if re.search(r"qwen[-_]?3", mid):
        return "qwen3"
    # Gemma 2 / 3
    if re.search(r"gemma[-_]?2", mid):
        return "gemma2"
    if re.search(r"gemma[-_]?3", mid):
        return "gemma3"
    return "other"


def clean_chat_template_tokens(text: str, model_id: str) -> str:
    """
    Limpia tokens de plantilla de chat de la salida del modelo para métricas (F1, ROUGE, etc.),
    respetando la familia de modelo (Llama 2/3, Qwen 2.5/3, Gemma 2/3).
    """
    if not text:
        return ""
    t = text
    family = detect_model_family(model_id)

    # --- Llama 3.x: <|begin_of_text|>, <|start_header_id|>role<|end_header_id|>, <|eot_id|>, etc. ---
    if family == "llama3":
        t = re.sub(r"<\|begin_of_text\|>", " ", t)
        t = re.sub(r"<\|start_header_id\|>.*?<\|end_header_id\|>", " ", t, flags=re.DOTALL)
        t = re.sub(r"<\|eot_id\|>", " ", t)

    # --- Llama 2: <s> [/INST] <<SYS>> <</SYS>> ---
    if family == "llama2":
        t = re.sub(r"</?s>", " ", t)
        t = re.sub(r"\[INST\]|\[/INST\]|<<SYS>>|<</SYS>>", " ", t)

    # --- Qwen ChatML (2.5 / 3): <|im_start|>role, <|im_end|>, <think>...</think> ---
    if family.startswith("qwen"):
        t = re.sub(r"<think>.*?</think>", " ", t, flags=re.DOTALL | re.IGNORECASE)
        t = re.sub(r"<\|im_start\|>.*?\n", " ", t, flags=re.DOTALL)
        t = re.sub(r"<\|im_end\|>", " ", t)

    # --- Gemma 2/3: <bos>, <start_of_turn>role, <end_of_turn> ---
    if family.startswith("gemma"):
        t = re.sub(r"<bos>", " ", t)
        t = re.sub(r"<start_of_turn>.*?\n", " ", t, flags=re.DOTALL)
        t = re.sub(r"<end_of_turn>", " ", t)

    # Limpieza genérica de tokens <|...|> residuales
    t = re.sub(r"<\|[^>]*\|>", " ", t)
    # Normalizar espacios
    t = re.sub(r"\s+", " ", t).strip()
    return t


def apply_chat_template_to_prompt(
    prompt: str,
    tokenizer,
    system_prompt: str = "",
) -> str:
    """
    Envuelve un prompt de usuario en la chat_template del tokenizer (si existe).

    Asumimos un solo turno:
        system: <system_prompt> (opcional)
        user:   <prompt>
        assistant: <respuesta>
    """
    if not prompt:
        return ""
    if not hasattr(tokenizer, "apply_chat_template"):
        return prompt

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        # versiones antiguas pueden no aceptar add_generation_prompt
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            return prompt
    except Exception:
        return prompt


def apply_chat_template_to_dataset(
    ds: Dataset,
    tokenizer,
    system_prompt: str = "",
    num_proc: int | None = None,
) -> Dataset:
    """
    Reescribe la columna 'prompt' del dataset usando la chat_template (si existe).
    No toca 'chosen' ni 'rejected' (se asumen textos del assistant).
    """
    if ds is None:
        return None
    if not hasattr(tokenizer, "apply_chat_template"):
        return ds

    def _map_fn(batch):
        prompts = batch["prompt"]
        templated = [
            apply_chat_template_to_prompt(p, tokenizer, system_prompt=system_prompt)
            for p in prompts
        ]
        return {"prompt": templated}

    if num_proc is None:
        try:
            num_proc = os.cpu_count() or 1
        except Exception:
            num_proc = 1

    return ds.map(_map_fn, batched=True, num_proc=num_proc)

# ---------------------------------
# Semillas / reproducibilidad
# ---------------------------------
def set_global_seed(seed: int = 42) -> None:
    """
    Aplica la semilla a:
      - PYTHONHASHSEED (hash aleatorio de Python)
      - random, numpy
      - torch (CPU y CUDA, si está disponible)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

# ---------------------------------
# Callback para imprimir pérdidas por epoch
# ---------------------------------
class EpochLossPrinter(TrainerCallback):
    def __init__(self, method: str, beta: float):
        self.method = method
        self.beta = beta
        self.epoch_losses = defaultdict(list)
        self.prev_mean = None
        # Último loss de train y de validación por época
        self.last_train_loss = {}
        self.last_eval_loss = {}
        # Lista de accuracies de entrenamiento por época (para media por epoch)
        self.epoch_pairwise_acc = defaultdict(list)
        # Último accuracy de train visto en la época
        self.last_train_pairwise_acc = {}
        # Último accuracy de validación por época (viene de eval)
        self.last_eval_pairwise_acc = {}   

    def _epoch_idx(self, state):
        """
        Mapea el valor fraccional de state.epoch a un índice entero de época,
        usando ceil:
          0 < epoch <= 1  -> 1
          1 < epoch <= 2  -> 2
        """
        if state is None or state.epoch is None:
            return None
        return int(np.ceil(float(state.epoch)))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        ep = self._epoch_idx(state)
        if ep is None:
            return

        is_eval_log = any(k.startswith("eval_") for k in logs.keys())        

        # ---------------------------
        # 1) Pérdida de entrenamiento
        # ---------------------------
        loss = logs.get("loss", None)
        if loss is None:  # algunos trainers nombran distinta la métrica
            for k in ("train_loss", "dpo_loss", "orpo_loss"):
                if k in logs:
                    loss = logs[k]
                    break

        if loss is not None:
            loss = float(loss)
            # guardamos todos para el mean_loss
            self.epoch_losses[ep].append(loss)
            # y además el último loss de train de la época
            # (ignoramos logs de evaluación que traen claves "eval_*")
            if not is_eval_log:
                self.last_train_loss[ep] = loss

        # -------------------------
        # 2) Pérdida de validación
        # -------------------------
        eval_loss = logs.get("eval_loss", None)
        if eval_loss is not None:
            self.last_eval_loss[ep] = float(eval_loss)

        # -------------------------
        # 3) Pairwise accuracy (agnóstica a DPO/ORPO)
        # -------------------------
        # En train, TRL/Unsloth suele loguear "rewards/accuracies".
        pair_acc = None
        for k in ("rewards/accuracies", "pairwise_accuracy", "train_pairwise_accuracy"):
            if k in logs:
                pair_acc = logs[k]
                break

        if pair_acc is not None and not is_eval_log:
            pair_acc = float(pair_acc)
            self.epoch_pairwise_acc[ep].append(pair_acc)
            self.last_train_pairwise_acc[ep] = pair_acc


    def on_epoch_end(self, args, state, control, **kwargs):
        ep = self._epoch_idx(state)
        if ep is None:
            return

        vals = self.epoch_losses.get(ep, [])
        if not vals:
            print(f"[epoch {ep}] Sin logs de pérdida.", flush=True)
            return

        mean_loss = sum(vals) / len(vals)
        if self.prev_mean is None:
            reduction_str = "—"
        else:
            reduction = self.prev_mean - mean_loss
            reduction_str = f"{reduction:.6f}"
        self.prev_mean = mean_loss

        if self.method == "dpo":
            loss_desc = "DPO loss: objetivo de preferencias par-a-par; minimiza -log σ(β·Δlogπ) (ref-free), β controla la rigidez."
        else:
            loss_desc = "ORPO loss: combina NLL con un término de odds-ratio sobre (chosen vs rejected), ponderado por β."

        train_last = self.last_train_loss.get(ep, None)
        val_last = self.last_eval_loss.get(ep, None)
        train_last_str = f"{train_last:.6f}" if train_last is not None else "NA"
        val_last_str = f"{val_last:.6f}" if val_last is not None else "NA"

        # Medias y últimos valores de pairwise accuracy
        train_pair_list = self.epoch_pairwise_acc.get(ep, [])
        train_pacc_mean = sum(train_pair_list) / len(train_pair_list) if train_pair_list else None
        train_pacc_str = f"{train_pacc_mean:.4f}" if train_pacc_mean is not None else "NA"


        print(
            f"[epoch {ep}] mean_train_loss={mean_loss:.6f} "
            f"| train_last_loss={train_last_str} "
            f"| train_pair_acc_mean={train_pacc_str} "
            f"| reducción_vs_prev={reduction_str} | β={self.beta} | {loss_desc}",
            flush=True,
        )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """
        Se llama al final de cada evaluación (en este script: una vez por época).
        Aquí registramos/confirmamos el eval_loss de esa época y
        volvemos a imprimir un resumen centrado en validación.
        """
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss", None)
        if eval_loss is None:
            return

        ep = self._epoch_idx(state)
        if ep is None:
            return

        self.last_eval_loss[ep] = float(eval_loss)
        tr_last = self.last_train_loss.get(ep, None)
        tr_str = f"{tr_last:.6f}" if tr_last is not None else "NA"
        
        # Pairwise accuracy de validación (mismo key para DPO y ORPO en TRL/Unsloth)
        eval_pair = metrics.get("eval_rewards/accuracies", None)
        if eval_pair is None:
            eval_pair = metrics.get("eval_pairwise_accuracy", None)
        if eval_pair is not None:
            self.last_eval_pairwise_acc[ep] = float(eval_pair)
        eval_pair_str = f"{float(eval_pair):.4f}" if eval_pair is not None else "NA"

        print(
            f"[epoch {ep}] [val] eval_loss={float(eval_loss):.6f} "
            f"| train_last_loss={tr_str}",
            f"| val_pair_acc={eval_pair_str}",
            flush=True,
        )

# ---------------------------------
# Modelo + LoRA (Unsloth QLoRA)
# ---------------------------------
def build_model_tokenizer(model_id: str, preset: dict):
    model, tok = FastLanguageModel.from_pretrained(model_name=model_id,
        max_seq_length=preset["max_seq_len"],
        dtype=None,
        load_in_4bit=True,  # QLoRA
    )
    model = FastLanguageModel.get_peft_model(model,
        r=preset["lora_r"],
        lora_alpha=preset["lora_alpha"],
        lora_dropout=preset["lora_dropout"],
        bias="none",
        target_modules=preset["target_modules"],
        use_gradient_checkpointing="unsloth",
        max_seq_length=preset["max_seq_len"],
    )
    return model, tok
    
def export_base_merged_16bit(model_id: str, preset: dict, out_id: str) -> None:
    """
    Carga el modelo base (model_id) en 16-bit y lo guarda en:
        ./outputs/<out_id>/merged-16bit
    para luego convertirlo a GGUF con llama.cpp y usarlo en Ollama.
    """
    out_dir = os.path.join("./outputs", out_id)
    merged_dir = os.path.join(out_dir, "merged-16bit")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"[base_merged] Exportando modelo base '{model_id}' a {merged_dir} ...", flush=True)

    # 16-bit "normal" (no QLoRA, sin load_in_4bit)
    base_model, base_tok = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=preset["max_seq_len"],
        dtype=None,            # auto: bf16 si está soportado, si no fp16
        load_in_4bit=False,
    )

    base_model.save_pretrained(merged_dir)
    base_tok.save_pretrained(merged_dir)

    # Liberar memoria
    try:
        del base_model
    except Exception:
        pass
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    print(
        "[base_merged] Listo. Ejemplo de conversión a GGUF y registro en Ollama:\n"
        f"  python3 llama.cpp/convert_hf_to_gguf.py --outfile outputs/{out_id}/gguf/{out_id} --outtype f16 outputs/{out_id}/merged-16bit\n"
        f"  ollama create {out_id} -f ./outputs/{out_id}/gguf/Modelfile",
        flush=True,
    )
    
# ---------------------------------
# Guardado (LoRA + merge 16-bit opcional)
# ---------------------------------
def save_all(trainer,
    tokenizer,
    out_dir: str,
    save_merged: bool = True,
    save_gguf: bool = False,
    gguf_quant_method: str = "q4_k_m",
    llama_cpp_convert_script=None,
    modelfiles_dir=None,
    model_family=None,
    ollama_bin=None,
    run_ollama_create=True,
) -> None:

    """
    Guarda:
      - Adaptadores LoRA en `out_dir`
      - (Opcional) modelo merged 16-bit en `out_dir/merged-16bit`
      - (Opcional) export GGUF listo para Ollama en `out_dir/gguf`
    """
    os.makedirs(out_dir, exist_ok=True)

    # LoRA adapters + tokenizer
    trainer.model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    # Merge 16-bit (para vLLM / HF normal)
    if save_merged:
        try:
            trainer.model.save_pretrained_merged(
                os.path.join(out_dir, "merged-16bit"),
                tokenizer,
                save_method="merged_16bit",
            )
        except Exception:
            from unsloth import FastLanguageModel as _FLM

            merged = _FLM.merge_and_unload(trainer.model)
            merged.save_pretrained(os.path.join(out_dir, "merged-16bit"))
            tokenizer.save_pretrained(os.path.join(out_dir, "merged-16bit"))

    # GGUF / Ollama export usando la MISMA lógica 2-pasos que el modelo base:
    #   1) HF -> GGUF f16
    #   2) GGUF f16 -> GGUF cuantizado (q4_k_m, q8_0, ...)
    #   3) Modelfile + `ollama create`
    if save_gguf:
        try:
            merged_dir = os.path.join(out_dir, "merged-16bit")
            if not os.path.isdir(merged_dir):
                print(
                    f"[save][gguf] Directorio merged-16bit no encontrado: {merged_dir}. "
                    "No se puede convertir a GGUF/Ollama.",
                    flush=True,
                )
            else:
                if not llama_cpp_convert_script:
                    llama_cpp_convert_script = "llama.cpp/convert_hf_to_gguf.py"

                if not (modelfiles_dir and model_family):
                    print(
                        "[save][gguf] modelfiles_dir/model_family no especificados; "
                        "se omite conversión a GGUF/Ollama.",
                        flush=True,
                    )
                else:
                    # out_id = nombre lógico del modelo (último componente del path)
                    out_id = os.path.basename(os.path.normpath(out_dir))
                    print(
                        f"[save][gguf] Exportando finetuned '{out_id}' a GGUF + cuantizado + Ollama ...",
                        flush=True,
                    )

                    # Reutilizamos exactamente la misma función que para el modelo base
                    convert_hf_to_gguf_and_make_ollama(
                        out_id=out_id,
                        outtype=gguf_quant_method,
                        convert_script=llama_cpp_convert_script,
                        modelfiles_dir=modelfiles_dir,
                        model_family=model_family,
                        ollama_bin=ollama_bin or "ollama",
                        skip_if_exists=False,       # siempre recreamos el finetuned
                        ollama_name=out_id,         # nombre del modelo en Ollama
                    )
        except Exception as e:
            print(f"[save_all] Error en exportación GGUF/Ollama: {e}", flush=True)

def evaluate_golden_set_f1(model,
    tokenizer,
    golden_path: str = "golden_set_test.csv",
    model_id: str = "",
    run_type: str = "",
    method: str = "",
    results_dir: str = "evaluation_results",
    max_samples: int = 0,
    max_new_tokens: int = 128,
    batch_size: int = 4,
    use_chat_template: bool = False,
    system_prompt: str = "",
) -> None:
    """
    Evalúa el modelo sobre un golden set de QA usando F1 token-level
    entre la respuesta generada y la columna `reference_answer`, y
    guarda un CSV de resultados por pregunta.

    Parámetros:
        model: modelo (base o finetuned, PEFT/QLoRA).
        tokenizer: tokenizer correspondiente.
        golden_path: ruta al CSV con columnas `question` y `reference_answer`
                      (y opcionalmente `qid`).
        model_id: identificador del modelo a registrar en la columna `model_id`.
        run_type: etiqueta corta para el tipo de modelo ("base", "finetuned", etc.).
        method: "dpo" / "orpo" u otra etiqueta para log.
        results_dir: directorio donde se escribirá golden_f1_results.csv.
        max_samples: si >0, limita el nº de ejemplos evaluados.
        max_new_tokens: longitud máxima de generación por pregunta.
        batch_size: tamaño de batch para generación.

    Además:
      - Si use_chat_template=True, envuelve cada pregunta usando la
        chat_template configurada en el tokenizer (Llama 3.x, Qwen 2.5/3, Gemma 2/3, etc.).
      - Limpia tokens de plantilla de chat de la salida antes de calcular F1
        (y opcionalmente de reference_answer, por seguridad).
    """
    try:
        # Usamos la misma métrica que en rag_evaluation_simple.py
        from utils_custom_metrics import calculate_f1_score
    except Exception as e:
        print(f"[eval][golden] No se pudo importar utils_custom_metrics.calculate_f1_score: {e}")
        print("[eval][golden] Se omite evaluación F1 sobre golden set.")
        return

    path = Path(golden_path)
    if not path.exists():
        print(f"[eval][golden] Archivo golden set no encontrado: {path}. Se omite evaluación F1.")
        return

    print(f"[eval][golden] Cargando golden set desde: {path}")
    df = pl.read_csv(str(path), infer_schema_length=10000)
    needed = {"question", "reference_answer"}
    if not needed.issubset(set(df.columns)):
        print(f"[eval][golden] CSV debe contener columnas {needed}, contiene: {df.columns}")
        return

    # Nos quedamos con columnas relevantes; `qid` es opcional pero útil para trazabilidad
    cols = []
    if "qid" in df.columns:
        cols.append("qid")
    cols.extend(["question", "reference_answer"])
    df = df.select(cols)
    rows = df.to_dicts()

    if max_samples and max_samples > 0 and len(rows) > max_samples:
        rows = rows[:max_samples]

    if not rows:
        print("[eval][golden] Golden set vacío tras aplicar filtros. Nada que evaluar.")
        return

    # Preparar modelo para inferencia
    try:
        FastLanguageModel.for_inference(model)
    except Exception:
        pass

    if torch is not None:
        try:
            device = next(model.parameters()).device
        except Exception:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = None

    if device is not None and torch is not None:
        model.eval()

    total = len(rows)
    f1_scores: List[float] = []
    example_rows: List[Dict] = []
    eval_ts = datetime.now().isoformat(timespec="seconds")

    print(f"[eval][golden] Evaluando F1 sobre {total} ejemplos (batch_size={batch_size})...")

    for start in range(0, total, batch_size):
        batch = rows[start:start + batch_size]
        questions = [row["question"] for row in batch]

        # Aplicar plantilla de chat a cada pregunta si está activado
        if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            prompts = [
                apply_chat_template_to_prompt(q, tokenizer, system_prompt=system_prompt)
                for q in questions
            ]
        else:
            prompts = questions

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=getattr(getattr(model, "config", None), "max_position_embeddings", 2048),
        )

        if device is not None and torch is not None:
            enc = {k: v.to(device) for k, v in enc.items()}

        if torch is not None:
            with torch.no_grad():
                generated = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        else:
            generated = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )

        for i, row in enumerate(batch):
            # Longitud real del prompt (sin padding) para cortar la salida
            if torch is not None:
                input_len = int(enc["attention_mask"][i].sum().item())
            else:
                input_len = len(enc["input_ids"][i])

            gen_ids = generated[i][input_len:]
            raw_text = tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()

            # Limpieza específica de tokens de chat_template según familia de modelo
            gen_text = clean_chat_template_tokens(raw_text, model_id or "")
            ref_answer = clean_chat_template_tokens(row["reference_answer"] or "", model_id or "")

            try:
                f1 = float(calculate_f1_score(gen_text, ref_answer))
            except TypeError:
                # Fallback por si la implementación devuelve vector/lista
                val = calculate_f1_score(gen_text, ref_answer)
                f1 = float(val[0] if isinstance(val, (list, tuple)) else val)

            f1_scores.append(f1)

            # ID estable de la pregunta (usa qid si está, si no, índice global)
            qid_val = row.get("qid")
            if qid_val is None:
                qid_val = str(start + i)

            example_rows.append(
                {
                    "qid": str(qid_val),
                    "question": row["question"],
                    "reference_answer": ref_answer,
                    "generated_answer": gen_text,
                    "f1_score": f1,
                    "model_id": model_id or "",
                    "run_type": run_type or "",
                    "method": method or "",
                    "dataset": path.name,
                    "eval_timestamp": eval_ts,
                }
            )

    if not f1_scores:
        print("[eval][golden] No se pudieron calcular F1 scores.")
        return

    mean_f1 = float(sum(f1_scores) / len(f1_scores))
    print(f"[eval][golden] F1 promedio (generated vs reference_answer) en golden set: {mean_f1:.4f}")

    # Guardar resultados detallados en CSV, compatible a nivel de columnas de generación
    try:
        results_dir_path = Path(results_dir)
        results_dir_path.mkdir(parents=True, exist_ok=True)
        out_file = results_dir_path / "golden_f1_results.csv"

        new_df = pl.DataFrame(example_rows)
        if out_file.exists():
            try:
                prev_df = pl.read_csv(str(out_file), infer_schema_length=10000)
                combined = pl.concat([prev_df, new_df], how="vertical_relaxed")
            except Exception:
                combined = new_df
        else:
            combined = new_df

        combined.write_csv(str(out_file))
        print(f"[eval][golden] Resultados por pregunta guardados en: {out_file}")
    except Exception as e:
        print(f"[eval][golden] Error guardando golden_f1_results.csv: {e}")

# ---------------------------------
# Funciones helper para Ollama
# ---------------------------------
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Ejecuta un comando y retorna código de salida, stdout y stderr."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout or "", res.stderr or ""

def wait_for_file(pattern: str, timeout: int = 60, poll_interval: float = 2.0) -> str:
    """
    Espera hasta `timeout` segundos a que exista al menos un archivo que
    cumpla el patrón `pattern` (glob). Devuelve la primera ruta encontrada.

    Ejemplos:
        wait_for_file("./outputs/foo/gguf/foo-f16*.gguf")
        wait_for_file("./outputs/foo/gguf/foo-Q4_K_M.gguf")
    """
    deadline = time.time() + timeout
    last_matches = []
    while time.time() < deadline:
        matches = glob.glob(pattern)
        if matches:
            matches.sort()
            return matches[0]
        last_matches = matches
        time.sleep(poll_interval)

    raise RuntimeError(
        f"[wait_for_file] No se encontró ningún archivo para pattern={pattern} "
        f"tras {timeout} segundos. Últimos matches: {last_matches}"
    )

def ollama_model_exists(name: str, ollama_bin: str = "ollama") -> bool:
    """Verifica si un modelo existe en Ollama mediante 'ollama list'."""
    code, out, err = run_cmd([ollama_bin, "list"])
    if code != 0:
        print(f"[ollama] No se pudo listar modelos: {err}")
        return False
    # 'ollama list' retorna líneas con 'name:tag ...'
    return any(
        line.strip().startswith(f"{name}:") or line.strip().split()[0] == name
        for line in out.splitlines()
    )


# ---------------------------------
# Funciones helper para Ollama
# ---------------------------------
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Ejecuta un comando y retorna código de salida, stdout y stderr."""
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode, res.stdout or "", res.stderr or ""


def wait_for_file(pattern: str, timeout: int = 60, poll_interval: float = 2.0) -> str:
    """
    Espera hasta `timeout` segundos a que exista al menos un archivo que
    cumpla el patrón `pattern` (glob). Devuelve la primera ruta encontrada.

    Ejemplos:
        wait_for_file("./outputs/foo/gguf/foo-f16*.gguf")
        wait_for_file("./outputs/foo/gguf/foo-Q4_K_M.gguf")
    """
    deadline = time.time() + timeout
    last_matches = []
    while time.time() < deadline:
        matches = glob.glob(pattern)
        if matches:
            matches.sort()
            return matches[0]
        last_matches = matches
        time.sleep(poll_interval)

    raise RuntimeError(
        f"[wait_for_file] No se encontró ningún archivo para pattern={pattern} "
        f"tras {timeout} segundos. Últimos matches: {last_matches}"
    )


def ollama_model_exists(name: str, ollama_bin: str = "ollama") -> bool:
    """Verifica si un modelo existe en Ollama mediante 'ollama list'."""
    code, out, err = run_cmd([ollama_bin, "list"])
    if code != 0:
        print(f"[ollama] No se pudo listar modelos: {err}")
        return False
    # 'ollama list' retorna líneas con 'name:tag ...'
    return any(
        line.strip().startswith(f"{name}:") or line.strip().split()[0] == name
        for line in out.splitlines()
    )


def convert_hf_to_gguf_and_make_ollama(
    out_id: str,
    outtype: str,                 # método de cuantización (Q4_K_M, Q5_K_M, etc.)
    convert_script: str,
    modelfiles_dir: str,
    model_family,
    ollama_bin: str = "ollama",
    skip_if_exists: bool = True,
    ollama_name = None,
) -> None:
    """
    Prepara un modelo HF merged-16bit para Ollama en dos pasos:

      1) HF merged-16bit -> GGUF f16   (convert_hf_to_gguf.py)
      2) GGUF f16 -> GGUF cuantizado   (llama-quantize, p.ej. Q4_K_M)

    Luego genera el Modelfile y hace 'ollama create'.
    """
    out_dir   = os.path.join("./outputs", out_id)
    merged_dir = os.path.join(out_dir, "merged-16bit")
    gguf_dir   = os.path.join(out_dir, "gguf")
    os.makedirs(gguf_dir, exist_ok=True)

    model_name = ollama_name or os.path.basename(out_dir)

    if skip_if_exists and ollama_model_exists(model_name, ollama_bin=ollama_bin):
        print(f"[gguf][ollama] Ya existe '{model_name}' en Ollama. Se omite conversión/registro.")
        return

    if not os.path.isdir(merged_dir):
        raise RuntimeError(f"[gguf] No existe {merged_dir} para {out_id}. Ejecuta el merge 16-bit primero.")

    python_bin = sys.executable or "python3"

    # ------------------------------------------------------------------
    # 1) HF -> GGUF f16 (convert_hf_to_gguf.py)
    #    Usamos sufijo -f16 para que el cuantizado use <model_name>.gguf
    # ------------------------------------------------------------------
    outfile_prefix_f16 = os.path.join(gguf_dir, model_name + "-f16")
    cmd = [
        python_bin,
        convert_script,
        "--outfile", outfile_prefix_f16,
        "--outtype", "f16",
        merged_dir,
    ]
    print(f"[gguf] Ejecutando (HF -> GGUF f16): {' '.join(cmd)}")
    code, out, err = run_cmd(cmd)
    if out:
        print(out)
    if err:
        # Lo mostramos siempre para poder depurar si algo raro pasa
        print(f"[gguf][convert_hf_to_gguf stderr]:\n{err}")
    if code != 0:
        raise RuntimeError(f"[gguf] convert_hf_to_gguf.py falló (code={code})")

    # Esperar hasta 60s a que aparezca el GGUF f16 (con o sin sufijo)
    f16_pattern = outfile_prefix_f16 + "*"
    f16_gguf = wait_for_file(f16_pattern, timeout=60, poll_interval=2.0)
    print(f"[gguf] GGUF f16 localizado en: {f16_gguf}")

    # ------------------------------------------------------------------
    # 2) Quantize: f16 -> Q4_K_M (u otro método de llama-quantize)
    # ------------------------------------------------------------------
    quant_method = (outtype or "Q4_K_M").upper()
    quantized_gguf = os.path.join(gguf_dir, f"{model_name}.gguf")

    # Ajusta la ruta si llama-quantize no está en el PATH:
    quant_cmd = ["./llama.cpp/build/bin/llama-quantize", f16_gguf, quantized_gguf, quant_method]
    
    print(f"[gguf][quantize] Ejecutando: {' '.join(quant_cmd)}")
    code_q, out_q, err_q = run_cmd(quant_cmd)
    if out_q:
        print(f"[gguf][quantize] Cuantizando a {outtype}")
    if err_q:
        print(f"[gguf][quantize stderr]:\n{err_q}")
    if code_q != 0:
        raise RuntimeError(f"[gguf][quantize] llama-quantize falló (code={code_q})")

    # Esperar hasta 60s a que exista el GGUF cuantizado exacto
    quantized_gguf = wait_for_file(quantized_gguf, timeout=60, poll_interval=2.0)
    print(f"[gguf][quantize] GGUF cuantizado disponible en: {quantized_gguf}")

    # ------------------------------------------------------------------
    # 3) Modelfile
    # ------------------------------------------------------------------
    if model_family:
        src_modelfile = os.path.join(modelfiles_dir, model_family)
        dst_modelfile = os.path.join(gguf_dir, "Modelfile")
        with open(src_modelfile, "r", encoding="utf-8") as f:
            content = f.read()
        # El Modelfile suele tener algo como: model ./Qwen2.5-3B-Instruct.gguf
        content = content.replace("[model_name]", os.path.basename(quantized_gguf))
        with open(dst_modelfile, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[gguf] Modelfile generado en {dst_modelfile}")
    else:
        print("[gguf] Aviso: --model_family no establecido. Debes proveer un Modelfile manualmente.")

    # ------------------------------------------------------------------
    # 4) ollama create
    # ------------------------------------------------------------------
    if ollama_bin:
        cmd_ollama = [
            ollama_bin,
            "create",
            model_name,
            "-f",
            os.path.join(gguf_dir, "Modelfile"),
        ]
        print(f"[gguf][ollama] Ejecutando: {' '.join(cmd_ollama)}")
        code_o, out_o, err_o = run_cmd(cmd_ollama)
        if out_o:
            print(out_o)
        if err_o:
            print(f"[gguf][ollama stderr]:\n{err_o}")
        if code_o != 0:
            raise RuntimeError(f"[gguf][ollama] ollama create falló (code={code_o})")
        print(f"[gguf][ollama] Modelo registrado en Ollama como '{model_name}'.")
    else:
        print("[gguf][ollama] ollama_bin no especificado; se omite 'ollama create'.")



def evaluate_golden_set_f1_ollama(
    model_name: str,
    golden_path: str,
    model_id: str = "",
    run_type: str = "",
    method: str = "",
    results_dir: str = "evaluation_results",
    max_samples: int = 0,
    max_new_tokens: int = 128,
    system_prompt: str = "",
) -> None:
    """
    Evalúa un modelo en Ollama (base o finetuned) sobre un golden set de QA y
    calcula F1 token-level usando utils_custom_metrics.calculate_f1_score.

    Parámetros
    ----------
    model_name: nombre del modelo en Ollama (ej. 'base-Qwen2.5-3B-Instruct').
    golden_path: ruta al CSV con columnas `question` y `reference_answer`
                 (y opcionalmente `qid`).
    model_id: id lógico del modelo (para logging en CSV; suele ser el HF id).
    run_type: etiqueta ('base-ollama', 'ft-ollama', etc.).
    method: 'dpo', 'orpo' o etiqueta que quieras loggear.
    results_dir: carpeta donde se escribe golden_f1_results.csv.
    max_samples: si >0, recorta el número de filas evaluadas.
    max_new_tokens: no se aplica directamente, se mantiene por simetría.
    system_prompt: se pasa como system_message al Agent de Agno (mensaje de sistema).
    """
    try:
        from utils_custom_metrics import calculate_f1_score
    except Exception as e:
        print(f"[eval][ollama] Sin utils_custom_metrics: {e}. Se omite evaluación.")
        return

    path = Path(golden_path)
    if not path.exists():
        print(f"[eval][ollama] Golden set no encontrado: {path}")
        return

    print(f"[eval][ollama] Cargando golden set desde: {path}")
    df = pl.read_csv(str(path), infer_schema_length=10000)
    needed = {"question", "reference_answer"}
    if not needed.issubset(set(df.columns)):
        print(f"[eval][ollama] CSV debe tener columnas {needed}, contiene: {df.columns}")
        return

    # Selección de columnas mínimas
    cols = []
    if "qid" in df.columns:
        cols.append("qid")
    cols.extend(["question", "reference_answer"])
    df = df.select(cols)
    rows = df.to_dicts()

    if max_samples and len(rows) > max_samples:
        rows = rows[:max_samples]

    if not rows:
        print("[eval][ollama] Golden set vacío tras filtros. Nada que evaluar.")
        return

    # -------- Backend de generación: Agno Agent(Ollama) --------
    agno_agent = None
    use_agno = False

    try:
        from agno.agent import Agent
        from agno.models.ollama import Ollama as AgnoOllama

        # system_prompt se inyecta como verdadero mensaje de sistema
        agno_agent = Agent(
            model=AgnoOllama(id=model_name),
            system_message=system_prompt or None,
            show_tool_calls=False,
            markdown=False,
        )
        use_agno = True
        print("[eval][ollama] Usando Agno Agent + Ollama para generación.")
    except Exception as e:
        print(f"[eval][ollama] Agno no disponible ({e}). Se omite evaluación vía Ollama.")
        return

    def _gen_agno(question: str) -> str:
        try:
            # La pregunta se pasa limpia como mensaje de usuario;
            # el framing viene del system_message.
            resp = agno_agent.run(question.strip(), stream=False)
            return (str(resp.content) if hasattr(resp, "content") else str(resp)).strip()
        except Exception as e:
            print(f"[eval][ollama] Error usando Agno Agent: {e}")
            return ""

    # ----------------- Bucle principal de evaluación -----------------
    f1_scores = []
    example_rows = []
    eval_ts = datetime.now().isoformat(timespec="seconds")

    for idx, row in enumerate(rows):
        q = row["question"]
        ref = row["reference_answer"] or ""

        gen_text = _gen_agno(q) if use_agno else ""

        # En Agno + Ollama ya no esperamos tokens de plantilla tipo ChatML;
        # solo normalizamos espacios para robustez de F1.
        gen_text_clean = re.sub(r"\s+", " ", (gen_text or "").strip())
        ref_clean = re.sub(r"\s+", " ", (ref or "").strip())

        try:
            f1 = float(calculate_f1_score(gen_text_clean, ref_clean))
        except Exception:
            f1 = 0.0

        f1_scores.append(f1)
        qid_val = row.get("qid", idx)
        example_rows.append({
            "qid": str(qid_val),
            "question": q,
            "reference_answer": ref_clean,
            "generated_answer": gen_text_clean,
            "f1_score": f1,
            "model_id": model_id or model_name,
            "run_type": run_type or "base-ollama",
            "method": method or "",
            "dataset": path.name,
            "eval_timestamp": eval_ts,
        })

    if not f1_scores:
        print("[eval][ollama] Sin F1s computables.")
        return

    mean_f1 = sum(f1_scores) / len(f1_scores)
    print(f"[eval][ollama] F1 promedio: {mean_f1:.4f}")

    # Append/Write CSV
    try:
        results_dir_path = Path(results_dir)
        results_dir_path.mkdir(parents=True, exist_ok=True)
        out_file = results_dir_path / "golden_f1_results.csv"
        new_df = pl.DataFrame(example_rows)
        if out_file.exists():
            prev_df = pl.read_csv(str(out_file), infer_schema_length=10000)
            combined = pl.concat([prev_df, new_df], how="vertical_relaxed")
        else:
            combined = new_df
        combined.write_csv(str(out_file))
        print(f"[eval][ollama] Guardado en {out_file}")
    except Exception as e:
        print(f"[eval][ollama] Error guardando CSV: {e}")

# ---------------------------------
# CLI y main
# ---------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="QLoRA + Unsloth con DPO/ORPO sobre CSV (Polars).")
    ap.add_argument("--csv_path", required=True, help="Ruta al CSV con columnas prompt,chosen,rejected")
    ap.add_argument("--method", choices=["dpo","orpo"], required=True)
    ap.add_argument("--model_id", required=True, help="Modelo base (HF/Unsloth). Ej: unsloth/llama-3.2-3b-bnb-4bit")
    ap.add_argument("--output_model_id", default=None, help="Nombre del modelo de salida (si no se pasa, se autogenera con timestamp al finalizar)")
    ap.add_argument("--preset_index", type=int, default=0, help="Índice de preset a usar (sólo se aplica ese índice)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.0)
    ap.add_argument("--no_merge", action="store_true")
    ap.add_argument("--save_gguf",
        action="store_true",
        default=True,
        help="Si se activa, exporta el modelo finetuned a GGUF listo para Ollama (save_pretrained_gguf).",
    )
    ap.add_argument("--gguf_quant_method",
        type=str,
        default="Q4_K_M",
        help="Método de cuantización para llama-quantize (Q4_K_M, Q5_K_M, Q8_0, ...).",
    )
    ap.add_argument(
        "--llama_cpp_convert_script",
        type=str,
        default="llama.cpp/convert_hf_to_gguf.py",
        help="Ruta al script convert_hf_to_gguf.py de llama.cpp (se usa cuando --save_gguf está activo).",
    )
    ap.add_argument(
        "--modelfiles_dir",
        type=str,
        default="modelfiles",
        help="Directorio donde se encuentran los Modelfiles base por familia (para Ollama).",
    )
    ap.add_argument(
        "--model_family",
        type=str,
        default=None,
        help="Nombre de archivo dentro de --modelfiles_dir para usar como Modelfile base (ej. 'qwen2_5'). Obligatorio si --save_gguf.",
    )
    ap.add_argument(
        "--ollama_create",
        action="store_true",
        help="Si se activa junto con --save_gguf, ejecuta 'ollama create <output_model_id>' tras generar el GGUF.",
    )
    ap.add_argument(
        "--ollama_bin",
        type=str,
        default="ollama",
        help="Ruta o nombre del binario de Ollama (default: 'ollama' en PATH).",
    )
    
    ap.add_argument(
        "--export_base_merged",
        action="store_true",
        help="Si se activa, exporta también el modelo base (--model_id) en HF 16-bit a outputs/<base_output_model_id>/merged-16bit.",
    )
    ap.add_argument(
        "--base_output_model_id",
        type=str,
        default=None,
        help="Nombre de carpeta para exportar el modelo base merged-16bit. Si se omite, se deriva de --model_id.",
    )
    
    ap.add_argument("--truncation_mode", choices=["keep_end","keep_start"], default="keep_end",
                     help="Estrategia de truncado para prompts/completions (algunos builds de Unsloth/TRL lo leen desde TrainingArguments).")    
    ap.add_argument("--disable_dropout", action="store_true", help="Desactiva capas de dropout durante el entrenamiento (campo esperado por Unsloth ORPOTrainer).")
    ap.add_argument("--dataset_num_proc", type=int, default=None,
                    help="Nº de procesos para datasets.map. Si no se especifica, se infiere de os.cpu_count().")
    ap.add_argument("--seed", type=int, default=42,
                    help="Semilla global (random/NumPy/PyTorch/HF). Default=42.")
    ap.add_argument("--gpu", type=int, default=None,
                    help="Índice de GPU a usar (ej. 0 o 1). Si no se pasa, usa la selección por defecto.")
    ap.add_argument("--use_liger_loss", action="store_true",     
                    help="Activa la loss acelerada con liger-kernel en DPO/ORPO (requiere `pip install liger-kernel`).")
    ap.add_argument(
        "--precompute_ref_log_probs",
        action="store_true",
        default=False,
        help="Sólo útil si usas DPO con modelo de referencia (ref_model). "
             "Precomputa los log-probs del ref para acelerar el entrenamiento. "
             "En modo reference-free debe permanecer en False."
    )
    ap.add_argument("--use_logits_to_keep", action="store_true",
                    help="Activa retención de top-K logits por token para ahorrar memoria (compat Unsloth DPO).")
    ap.add_argument("--logits_to_keep", type=int, default=256,
                    help="K de logits a retener cuando --use_logits_to_keep está activo (p. ej. 256/512/1024).")

    ap.add_argument("--padding_free", action="store_true",
                    help="Activa batching padding-free si el trainer lo soporta (algunos builds de Unsloth/TRL lo leen desde TrainingArguments).")

    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Factor de suavizado de etiquetas que algunos builds de TRL/Unsloth esperan en TrainingArguments (default=0.0).")
                    
    ap.add_argument("--use_weighting", action="store_true",
                    help="Activa ponderación en DPO (algunos builds de UnslothDPOTrainer requieren args.use_weighting). "                         "Si no se pasa, se activa automáticamente si loss_weights != 1.0.")               

    # --- Evaluación opcional sobre golden_set_test.csv (F1 Score) ---
    ap.add_argument(
        "--golden_eval_path",
        type=str,
        default="golden_set_test.csv",
        help="Ruta al CSV de golden set con columnas 'question' y 'reference_answer' para evaluar F1 del modelo.",
    )
    ap.add_argument(
        "--golden_eval_max_samples",
        type=int,
        default=0,
        help="Máximo de ejemplos del golden set a evaluar (0 = todos).",
    )
    ap.add_argument(
        "--golden_eval_max_new_tokens",
        type=int,
        default=128,
        help="Máx. tokens nuevos a generar por pregunta durante la evaluación.",
    )
    ap.add_argument(
        "--golden_eval_batch_size",
        type=int,
        default=1,
        help="Tamaño de batch para evaluación en GPU/CPU.",
    )
    
    ap.add_argument(
        "--chat_system_prompt",
        type=str,
        default="",
        help="Mensaje de sistema opcional cuando se aplica chat_template.",
    )
    
    ap.add_argument("--base_to_ollama", action="store_true",
        help="Si se activa, prepara el modelo base en Ollama (HF->merged-16bit->GGUF Q4_K_M->ollama create).")
    ap.add_argument("--ollama_base_name", type=str, default=None,
        help="Nombre del modelo base en Ollama. Por defecto: base-<os.path.basename(model_id)>.")    
    ap.add_argument("--skip_ollama_if_exists", action="store_true",
        help="No ejecuta 'ollama create' si el modelo ya existe en 'ollama list'.")
    ap.add_argument("--eval_base_ollama", action="store_true",
        help="Si se activa junto con --base_to_ollama, evalúa F1 del modelo base cuantizado en Ollama sobre el golden set.")

    ap.add_argument(
        "--only_eval_ollama",
        action="store_true",
        help=(
            "Si se activa, NO entrena ni exporta modelos; solo ejecuta la "
            "evaluación F1 vía Ollama sobre el modelo base y/o el finetuned "
            "ya existentes en 'ollama list'."
        ),
    )

    return ap.parse_args()

def main():
    args = parse_args()
    
    
    # Resolver desde el inicio el nombre del modelo de salida para reutilizarlo
    if args.output_model_id is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        resolved_output_model_id = f"{args.method}-qlora-{os.path.basename(args.model_id)}-{ts}"
    else:
        resolved_output_model_id = args.output_model_id
    args.output_model_id = resolved_output_model_id

    
    if not (0 <= args.preset_index < len(PRESETS)):
        raise ValueError(f"preset_index fuera de rango. Opciones válidas: 0..{len(PRESETS)-1}")  
    
    preset = PRESETS[args.preset_index]
    print(f"[preset] Usando preset #{args.preset_index}: {preset}")
    
    # 0) Pinning de GPU antes de cargar el modelo
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"[device] Usando GPU index {args.gpu} (vía CUDA_VISIBLE_DEVICES).")

    # 0b) Semilla global
    set_global_seed(int(args.seed))

    # 0c) Preparación de nombre base en Ollama
    if getattr(args, "ollama_base_name", None):
        ollama_base_name = args.ollama_base_name
    else:
        safe_name = os.path.basename(args.model_id).replace("/", "_")
        ollama_base_name = f"base-{safe_name}"

    # 0c-bis) Modo solo evaluación vía Ollama (sin entrenamiento)
    if getattr(args, "only_eval_ollama", False):
        print(
            "[main] --only_eval_ollama activo: se omite entrenamiento/export "
            "y solo se ejecutan evaluaciones vía Ollama.",
            flush=True,
        )

        # Evaluación del modelo base en Ollama (si se pidió)
        if getattr(args, "eval_base_ollama", False) and getattr(args, "golden_eval_path", None):
            base_name = ollama_base_name
            if ollama_model_exists(base_name, ollama_bin=getattr(args, "ollama_bin", "ollama")):
                print(f"[eval][base-ollama] Evaluando modelo base existente '{base_name}' vía Ollama + Agno.")
                evaluate_golden_set_f1_ollama(
                    model_name=base_name,
                    golden_path=args.golden_eval_path,
                    model_id=base_name,
                    run_type="base-ollama",
                    method="-",  # base sin RLHF
                    results_dir="evaluation_results",
                    max_samples=getattr(args, "golden_eval_max_samples", 0),
                    max_new_tokens=getattr(args, "golden_eval_max_new_tokens", 128),
                    system_prompt=args.chat_system_prompt,
                )
            else:
                print(
                    f"[eval][base-ollama] Modelo base '{base_name}' no "
                    "encontrado en 'ollama list'. Se omite evaluación.",
                    flush=True,
                )

        # Evaluación del modelo finetuned cuantizado en Ollama
        if getattr(args, "golden_eval_path", None):
            ft_ollama_name = args.output_model_id
            if ollama_model_exists(ft_ollama_name, ollama_bin=getattr(args, "ollama_bin", "ollama")):
                print(f"[eval][ft-ollama] Evaluando modelo finetuned existente '{ft_ollama_name}' vía Ollama + Agno.")
                evaluate_golden_set_f1_ollama(
                    model_name=ft_ollama_name,
                    golden_path=args.golden_eval_path,
                    model_id=args.output_model_id,
                    run_type="ft-ollama",
                    method=args.method,
                    results_dir="evaluation_results",
                    max_samples=getattr(args, "golden_eval_max_samples", 0),
                    max_new_tokens=getattr(args, "golden_eval_max_new_tokens", 128),
                    system_prompt=args.chat_system_prompt,
                )
            else:
                print(
                    f"[eval][ft-ollama] Modelo finetuned '{ft_ollama_name}' no "
                    "encontrado en 'ollama list'. Se omite evaluación.",
                    flush=True,
                )

        return

    # 0d) Preparación del modelo base → GGUF Q4_K_M → Ollama (antes de entrenar)    
    
    
    if getattr(args, "base_to_ollama", False):
        already_exists = ollama_model_exists(ollama_base_name, ollama_bin=getattr(args, "ollama_bin", "ollama"))
        if already_exists and getattr(args, "skip_ollama_if_exists", False):
            print(f"[base][ollama] '{ollama_base_name}' ya existe. Se omite export/conversión.")
        else:
            # Exportar a merged-16bit si no existe
            base_out_id = args.base_output_model_id or ollama_base_name
            base_out_dir = os.path.join("./outputs", base_out_id)
            merged_dir = os.path.join(base_out_dir, "merged-16bit")

            if not os.path.isdir(merged_dir):
                print(f"[base][merge] Exportando modelo base a {merged_dir} ...")
                try:
                    export_base_merged_16bit(args.model_id, preset, base_out_id)
                except Exception as e:
                    print(f"[base][merge] No se pudo exportar HF 16-bit del base ({e}).")

            # Si hay merged-16bit, intentamos GGUF + Ollama (idempotente)
            if os.path.isdir(os.path.join(base_out_dir, "merged-16bit")):
                try:
                    convert_hf_to_gguf_and_make_ollama(out_id=base_out_id,
                    outtype=getattr(args, "gguf_quant_method", "Q4_K_M"),
                    convert_script=getattr(args, "llama_cpp_convert_script", "llama.cpp/convert_hf_to_gguf.py"),
                    modelfiles_dir=getattr(args, "modelfiles_dir", "modelfiles"),
                    model_family=getattr(args, "model_family", None),
                    ollama_bin=getattr(args, "ollama_bin", "ollama"),
                    skip_if_exists=bool(getattr(args, "skip_ollama_if_exists", False)),
                    ollama_name=ollama_base_name,
                    )
                except Exception as e:
                    print(f"[base][gguf/ollama] Error preparando base en Ollama: {e}")
    
    # 0e) (Opcional) exportar modelo base en 16-bit HF antes de entrenar QLoRA (modo legacy)
    if getattr(args, "export_base_merged", False) and not getattr(args, "base_to_ollama", False):
        base_out_id = args.base_output_model_id
        if not base_out_id:
            # Derivamos algo tipo base-Qwen2.5-3B-Instruct-bnb-4bit
            safe_name = os.path.basename(args.model_id).replace("/", "_")
            base_out_id = f"base-{safe_name}"
        print(f"[base_merged] Exportando modelo base a outputs/{base_out_id}/merged-16bit ...")
        export_base_merged_16bit(args.model_id, preset, base_out_id)

    # 1) Datos (Polars)
    df = read_and_preprocess(args.csv_path)
    tr, va, te = split_df_hash(df, train=args.train_ratio, val=args.val_ratio, test=args.test_ratio)
    print(f"[data] train={tr.height}, val={va.height}, test={te.height}")

    # 2) HF datasets para TRL
    ds_train = to_hf_dataset(tr)
    ds_test = to_hf_dataset(te) if te.height > 0 else None
    ds_val = to_hf_dataset(va) if va.height > 0 else None

    # Si hay validación, activamos evaluación por epoch
    _eval_strategy = "epoch" if (va.height > 0) else "no"

    # 3) Modelo QLoRA
    model, tok = build_model_tokenizer(args.model_id, preset)
    
    # 3a) Aplicar chat_template a los prompts del dataset (obligatorio si el tokenizer lo soporta)
    if hasattr(tok, "apply_chat_template"):
        print("[chat_template] Aplicando chat_template a prompts de train/val/test...")
        ds_train = apply_chat_template_to_dataset(
            ds_train,
            tok,
            system_prompt=args.chat_system_prompt,
            num_proc=getattr(args, "dataset_num_proc", None),
        )
        if ds_val is not None:
            ds_val = apply_chat_template_to_dataset(
                ds_val,
                tok,
                system_prompt=args.chat_system_prompt,
                num_proc=getattr(args, "dataset_num_proc", None),
            )
        if ds_test is not None:
            ds_test = apply_chat_template_to_dataset(
                ds_test,
                tok,
                system_prompt=args.chat_system_prompt,
                num_proc=getattr(args, "dataset_num_proc", None),
            )
    else:
        print("[chat_template] El tokenizer no expone apply_chat_template; se continúa con prompts planos.")

    # 3b) Evaluación opcional del modelo base en el golden set (antes de entrenar)
    # COMENTADO: Solo se evalúa vía Ollama+Agno (paso 9)
    # if getattr(args, "golden_eval_path", None):
    #     print("[eval][golden] Evaluando modelo base (sin RLHF) antes del entrenamiento...")
    #     evaluate_golden_set_f1(
    #         model=model,
    #         tokenizer=tok,
    #         golden_path=args.golden_eval_path,
    #         model_id=args.model_id,
    #         run_type="base",
    #         method=args.method,
    #         results_dir="evaluation_results",
    #         max_samples=getattr(args, "golden_eval_max_samples", 0),
    #         max_new_tokens=getattr(args, "golden_eval_max_new_tokens", 128),
    #         batch_size=getattr(args, "golden_eval_batch_size", 4),
    #         use_chat_template=True,
    #         system_prompt=args.chat_system_prompt,
    #     )

    # 4) TrainingArguments comunes
    train_args = TrainingArguments(
        output_dir=f"./outputs/{args.output_model_id or 'tmp'}",
        per_device_train_batch_size=preset["per_device_bs"],
        gradient_accumulation_steps=preset["grad_accum"],
        learning_rate=preset["lr_dpo"] if args.method == "dpo" else preset["lr_orpo"],
        warmup_ratio=preset["warmup_ratio"],
        num_train_epochs=args.epochs,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        seed=int(args.seed),
        data_seed=int(args.seed),
        save_steps=999999,           # evitamos checkpoints intermedios
        save_total_limit=1,
        remove_unused_columns=False,
        optim="adamw_8bit",
        report_to=[],

    )
    
    # --- Compatibilidad Unsloth/TRL ---

    
    # 1) ORPO/DPO Trainer espera que exista model_init_kwargs.
    #    Debe ser None si ya pasamos `model=` instanciado.
    if not hasattr(train_args, "model_init_kwargs"):
        setattr(train_args, "model_init_kwargs", None)
    else:
        train_args.model_init_kwargs = None
    # --- Fix para DPO: algunos builds leen args.ref_model_init_kwargs / ref_model_kwargs
    if not hasattr(train_args, "ref_model_init_kwargs"):
        setattr(train_args, "ref_model_init_kwargs", None)
    else:
        train_args.ref_model_init_kwargs = None
    if not hasattr(train_args, "ref_model_kwargs"):
        setattr(train_args, "ref_model_kwargs", None)
    else:
        train_args.ref_model_kwargs = None
        
    # Algunos builds de Unsloth/TRL esperan nombres de adaptadores PEFT:
    #     - model_adapter_name: el adaptador activo del modelo principal (PEFT).
    #     - ref_adapter_name:   el adaptador del modelo de referencia (si existiera).
    #     Dado que usamos PEFT con nombre por defecto, fijamos "default". Para el ref (ref-free), None.
    if not hasattr(train_args, "model_adapter_name"):
        setattr(train_args, "model_adapter_name", "default")
    else:
        train_args.model_adapter_name = "default"
    if not hasattr(train_args, "ref_adapter_name"):
        setattr(train_args, "ref_adapter_name", None)
    else:
        train_args.ref_adapter_name = None

    # Atributos que algunos builds de Unsloth DPOTrainer leen desde TrainingArguments
    # - ref_model_init_kwargs: debe existir si el trainer lo consulta (aún si es None)
    # - reference_free: obligatorio cuando entrenas DPO sin modelo de referencia (ref_model=None)
    # - model_adapter_name / ref_adapter_name: algunos builds los revisan para PEFT
    _unsloth_compat = {
        "ref_model_init_kwargs": None,
        "reference_free": True if args.method == "dpo" else False,
        "model_adapter_name": "peft",
        "ref_adapter_name": "",
    }
    for _k, _v in _unsloth_compat.items():
        try:
            setattr(train_args, _k, _v)
        except Exception:
            pass

    # 2) Algunas builds consultan generate_during_eval (y a veces predict_with_generate).
    #    Añadimos defaults "seguros" para evitar AttributeError.
    if not hasattr(train_args, "generate_during_eval"):
        setattr(train_args, "generate_during_eval", False)
    # predict_with_generate es usado por ciertos entrenadores para evaluar con generación.
    # Lo dejamos en False para no introducir sobrecosto inesperado.
    if not hasattr(train_args, "predict_with_generate"):
        setattr(train_args, "predict_with_generate", False)
        
    # 3) Unsloth ORPOTrainer usa args.max_length / args.max_prompt_length en lugar de kwargs.
    #    Los inyectamos desde el preset para evitar AttributeError.
    if not hasattr(train_args, "max_length"):
        setattr(train_args, "max_length", preset["max_seq_len"])
    else:
        train_args.max_length = preset["max_seq_len"]
    if not hasattr(train_args, "max_prompt_length"):
        setattr(train_args, "max_prompt_length", preset["max_prompt_len"])
    else:
        train_args.max_prompt_length = preset["max_prompt_len"]

    # 4) Modo de truncado: algunos builds de Unsloth ORPOTrainer leen args.truncation_mode
    #    en lugar de kwargs. Exponemos el valor del CLI (default 'keep_end').
    if not hasattr(train_args, "truncation_mode"):
        setattr(train_args, "truncation_mode", args.truncation_mode)
    else:
        train_args.truncation_mode = args.truncation_mode    
        
    # 5) Algunos builds de Unsloth ORPOTrainer consultan:
    #    args.max_completion_length (y a veces max_new_tokens / eval_max_new_tokens).
    #    Definimos long. de completion = seq_len - prompt_len, con clamp a 1.
    _comp_len = int(max(1, preset["max_seq_len"] - preset["max_prompt_len"]))
    if not hasattr(train_args, "max_completion_length"):
        setattr(train_args, "max_completion_length", _comp_len)
    else:
        train_args.max_completion_length = _comp_len
    # Aliases comunes usados por algunos trainers al generar/evaluar
    if not hasattr(train_args, "max_new_tokens"):
        setattr(train_args, "max_new_tokens", _comp_len)
    else:
        train_args.max_new_tokens = _comp_len
    if not hasattr(train_args, "eval_max_new_tokens"):
        setattr(train_args, "eval_max_new_tokens", _comp_len)
    else:
        train_args.eval_max_new_tokens = _comp_len
       
    # 5) Campos adicionales que Unsloth 2025.10.x puede requerir explícitamente:
    #    - ref_model_init_kwargs (si pasamos model/ref_model ya instanciados, debe ser None)
    #    - reference_free (DPO suele ir en modo ref-free)
    #    - model_adapter_name / ref_adapter_name (peft por defecto)
    #    - use_liger_loss (si quieres usar kernels liger, activa el flag y ten `liger-kernel` instalado)
    extra_compat = {
        "ref_model_init_kwargs": None,
        "reference_free": True if args.method == "dpo" else False,
        "model_adapter_name": "peft",
        "ref_adapter_name": "",
        "use_liger_loss": bool(getattr(args, "use_liger_loss", False)),
        "precompute_ref_log_probs": bool(getattr(args, "precompute_ref_log_probs", False)),
        "use_logits_to_keep": bool(getattr(args, "use_logits_to_keep", False)),
        "logits_to_keep": int(getattr(args, "logits_to_keep", 256)),

    }
    for _k, _v in extra_compat.items():
        if not hasattr(train_args, _k):
            setattr(train_args, _k, _v)
        else:
            setattr(train_args, _k, _v)

    # 6) Algunos builds consultan args.disable_dropout
    #    Exponemos el valor proveniente del CLI (default False).
    if not hasattr(train_args, "disable_dropout"):
        setattr(train_args, "disable_dropout", args.disable_dropout)
    else:
        train_args.disable_dropout = args.disable_dropout
       
    # 7) label/pad/eos ids - algunos builds de Unsloth/TRL los leen desde TrainingArguments
    #    a) asegura pad_token en el tokenizer (muchos Llama no tienen pad por defecto)
    if getattr(tok, "pad_token_id", None) is None:
        # si no hay pad, reutiliza eos como pad (práctica común en Llama)
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        elif getattr(tok, "unk_token", None) is not None:
            tok.pad_token = tok.unk_token
    #    b) expone ids en TrainingArguments si faltan
    if not hasattr(train_args, "label_pad_token_id"):
        setattr(train_args, "label_pad_token_id", -100)
    else:
        train_args.label_pad_token_id = -100
    if not hasattr(train_args, "pad_token_id") and getattr(tok, "pad_token_id", None) is not None:
        setattr(train_args, "pad_token_id", tok.pad_token_id)
    elif getattr(tok, "pad_token_id", None) is not None:
        train_args.pad_token_id = tok.pad_token_id
    if not hasattr(train_args, "eos_token_id") and getattr(tok, "eos_token_id", None) is not None:
        setattr(train_args, "eos_token_id", tok.eos_token_id)
    elif getattr(tok, "eos_token_id", None) is not None:
        train_args.eos_token_id = tok.eos_token_id
    # (opcional) homogeniza el padding a la derecha para preferencia-training
    try: tok.padding_side = "right"
    except Exception: pass

    # 8) Algunos builds de Unsloth ORPOTrainer consultan args.padding_value
    #    (si no existe, lanzan AttributeError). Debe ser un entero válido.
    #    Usamos el pad_token_id del tokenizer; si faltase, caemos a 0.
    _pv = getattr(tok, "pad_token_id", None)
    if _pv is None:
        # debería haberse fijado arriba al forzar pad=eos/unk; por seguridad:
        _pv = 0
    _pv = int(_pv)
    if not hasattr(train_args, "padding_value"):
        setattr(train_args, "padding_value", _pv)
    else:
        train_args.padding_value = _pv
        
    # Algunos builds consultan args.padding_free (booleano).
    #     No existe en HF por defecto; lo exponemos desde el CLI (default False).
    _pf = bool(getattr(args, "padding_free", False))
    if not hasattr(train_args, "padding_free"):
        setattr(train_args, "padding_free", _pf)
    else:
        train_args.padding_free = _pf
        
    # 9) Algunos builds de Unsloth ORPO/DPO leen args.beta desde TrainingArguments
    #    (además de recibirlo como kwarg). Inyéctalo para evitar AttributeError.
    if not hasattr(train_args, "beta"):
        setattr(train_args, "beta", float(preset["beta"]))
    else:
        # Forzamos a float por seguridad
        train_args.beta = float(preset["beta"])
        
    # 10) Algunos builds usan args.dataset_num_proc en datasets.map(...)
    #     Definimos un valor por defecto seguro a partir de los cores.
    if args.dataset_num_proc is not None and args.dataset_num_proc > 0:
        _num_proc = int(args.dataset_num_proc)
    else:
        _cpus = os.cpu_count() or 2
        # deja 1 core libre; cap a 8 para evitar oversubscription
        _num_proc = max(1, min(8, _cpus - 1))
    if not hasattr(train_args, "dataset_num_proc"):
        setattr(train_args, "dataset_num_proc", _num_proc)
    else:
        train_args.dataset_num_proc = _num_proc

    # 11) Programación de evaluacion inyectado después de inicializar TrainingArguments
    try: setattr(train_args, "evaluation_strategy", _eval_strategy)
    except Exception: pass
    try: setattr(train_args, "eval_strategy", _eval_strategy)
    except Exception: pass
    try: setattr(train_args, "do_eval", _eval_strategy != "no")
    except Exception: pass
    
    # 12) # Algunos builds de Unsloth/TRL consultan label_smoothing en TrainingArguments.
    if not hasattr(train_args, "label_smoothing"):
        setattr(train_args, "label_smoothing", float(args.label_smoothing))
    else:
        train_args.label_smoothing = float(args.label_smoothing)
    
    train_args.label_smoothing = float(getattr(args, "label_smoothing", 0.0))
    # Unsloth DPO recientes también consultan estos; si ya los tienes, se reescriben igual:
    train_args.reference_free   = (args.method == "dpo")
    train_args.model_adapter_name = "peft"
    train_args.ref_adapter_name   = ""
    train_args.use_liger_loss     = bool(getattr(args, "use_liger_loss", False))
    train_args.precompute_ref_log_probs = bool(getattr(args, "precompute_ref_log_probs", False))
    train_args.use_logits_to_keep = bool(getattr(args, "use_logits_to_keep", False))
    train_args.logits_to_keep     = int(getattr(args, "logits_to_keep", 256))
    train_args.padding_free       = bool(getattr(args, "padding_free", False))
    train_args.beta               = float(preset["beta"])
    train_args.max_length         = int(preset["max_seq_len"])
    train_args.max_prompt_length  = int(preset["max_prompt_len"])
    train_args.max_completion_length = int(max(1, preset["max_seq_len"] - preset["max_prompt_len"]))
    train_args.generate_during_eval   = False
    train_args.predict_with_generate = False
    train_args.model_init_kwargs      = None
    train_args.ref_model_init_kwargs  = None
    train_args.ref_model_kwargs       = None
    # IDs de tokens y padding
    if getattr(tok, "pad_token_id", None) is None:
        tok.pad_token = getattr(tok, "eos_token", getattr(tok, "unk_token", None))
    train_args.label_pad_token_id = -100
    if getattr(tok, "pad_token_id", None) is not None:
        train_args.pad_token_id = int(tok.pad_token_id)
    if getattr(tok, "eos_token_id", None) is not None:
        train_args.eos_token_id = int(tok.eos_token_id)
    train_args.padding_value = int(getattr(tok, "pad_token_id", 0))
    # (opcional) imprimir para verificar que no se pierdan al crear el trainer
    print(f"[debug] label_smoothing={train_args.label_smoothing}, reference_free={train_args.reference_free}, padding_free={train_args.padding_free}, beta={train_args.beta}")

    # ---- Unsloth DPO extra knobs ----
    if args.method == "dpo":
        # loss_type debe ser lista; default a ["sigmoid"]
        try:
            lt = getattr(train_args, "loss_type", None)
            if lt is None:
                lt = ["sigmoid"]
            elif not isinstance(lt, list):
                lt = [lt]
            train_args.loss_type = lt
        except Exception:
            train_args.loss_type = ["sigmoid"]

        # loss_weights debe existir y matchear la longitud de loss_type
        try:
            lw = getattr(train_args, "loss_weights", None)
            if lw is None or (isinstance(lw, list) and len(lw) != len(train_args.loss_type)):
                lw = [1.0] * len(train_args.loss_type)
            # normalizamos a float
            train_args.loss_weights = [float(x) for x in lw]
        except Exception:
            train_args.loss_weights = [1.0] * len(train_args.loss_type)
            
        # Algunos builds requieren args.use_weighting; si no viene del CLI, inferimos por los pesos.
        try:
            if getattr(args, "use_weighting", False):
                uw = True
            else:
                uw = any(abs(w - 1.0) > 1e-9 for w in train_args.loss_weights)
            setattr(train_args, "use_weighting", bool(uw))
        except Exception:
            setattr(train_args, "use_weighting", False)

        # Debug corto para verificar configuración DPO específica
        try:
            print(f"[debug][dpo] loss_type={train_args.loss_type}, loss_weights={train_args.loss_weights}, use_weighting={train_args.use_weighting}")
        except Exception:
            pass

        # --- f-divergence knobs (algunas builds de Unsloth/TR L los requieren)
        # Valores seguros por defecto: reverse_kl con alpha=1.0 y peso=1.0
        try:
            if not hasattr(train_args, "f_divergence_type"):
                train_args.f_divergence_type = "reverse_kl"   # opciones comunes: "reverse_kl","kl","chi2","jsd"
            if not hasattr(train_args, "f_divergence_alpha"):
                train_args.f_divergence_alpha = 1.0           # a veces llamado simplemente "alpha" en forks
            if not hasattr(train_args, "f_divergence_weight"):
                train_args.f_divergence_weight = 1.0
            print(f"[debug][dpo] f_divergence_type={train_args.f_divergence_type}, "
                  f"alpha={train_args.f_divergence_alpha}, weight={train_args.f_divergence_weight}")
        except Exception:
            pass
            
        # --- Aliases que algunas versiones de UnslothDPOTrainer exigen explícitamente:
        #     - f_alpha_divergence_coef  (usado para construir f_divergence_params)
        #     - f_divergence_coef        (alias de weight en ciertos forks)
        try:
            setattr(train_args, "f_alpha_divergence_coef", float(train_args.f_divergence_alpha))
        except Exception:
            train_args.f_alpha_divergence_coef = 1.0
        try:
            setattr(train_args, "f_divergence_coef", float(train_args.f_divergence_weight))
        except Exception:
            train_args.f_divergence_coef = 1.0
        # (Opcional defensivo) algunos forks leen directamente un dict ya preparado.
        try:
            train_args.f_divergence_params = {
                "alpha": float(train_args.f_divergence_alpha),
                "weight": float(train_args.f_divergence_weight),
                "type": str(train_args.f_divergence_type),
            }
        except Exception:
            pass

        # --- Compat RPO knobs: algunos builds consultan rpo_alpha incluso en DPO.
        # Preset neutro: None (evita entrar a ramas de RPO).
        if not hasattr(train_args, "rpo_alpha"):
            setattr(train_args, "rpo_alpha", None)
        if not hasattr(train_args, "ld_alpha"):
            setattr(train_args, "ld_alpha", None)
        # (opcional) debug
        try:
            print(f"[debug][dpo] rpo_alpha={train_args.rpo_alpha}, ld_alpha={train_args.ld_alpha}")
        except Exception:
            pass
            
    # --- Compat extra para builds que esperan 'tools' (funcall/tools-augmented prompts)
    if not hasattr(train_args, "tools"):
        setattr(train_args, "tools", None)   # o [] si prefieres lista vacía
    else:
        train_args.tools = None
        
    # --- Compat DPO ref-model flags (aunque estamos en reference-free)
    # Algunos builds consultan estos campos sin chequear si ref_model es None.
    if not hasattr(train_args, "sync_ref_model"):
        setattr(train_args, "sync_ref_model", False)
    else:
        train_args.sync_ref_model = False
    if not hasattr(train_args, "ref_model_batch_size"):
        setattr(train_args, "ref_model_batch_size", 1)
    if not hasattr(train_args, "ref_model_device_map"):
        setattr(train_args, "ref_model_device_map", None)
    if not hasattr(train_args, "static_reference_model"):
        setattr(train_args, "static_reference_model", True)
    try:
        print(f"[debug][dpo] sync_ref_model={train_args.sync_ref_model}, "
              f"ref_model_batch_size={train_args.ref_model_batch_size}, "
              f"ref_model_device_map={train_args.ref_model_device_map}, "
              f"static_reference_model={train_args.static_reference_model}")
    except Exception:
        pass
    # -----------------------------------

    # 5) Callback de pérdidas por epoch
    cb = EpochLossPrinter(method=args.method, beta=preset["beta"])

    # 6) Trainer DPO/ORPO
    if args.method == "dpo":
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=train_args,
            beta=preset["beta"],
            train_dataset=ds_train,
            tokenizer=getattr(tok, "tokenizer", tok),
            max_length=preset["max_seq_len"],
            max_prompt_length=preset["max_prompt_len"],
            eval_dataset=ds_val,
            callbacks=[cb],
        )
    else:
        trainer = ORPOTrainer(model=model,
            args=train_args,
            tokenizer=getattr(tok, "tokenizer", tok),
            beta=preset["beta"],
            train_dataset=ds_train,
            max_length=preset["max_seq_len"],
            max_prompt_length=preset["max_prompt_len"],
            eval_dataset=ds_val,
            callbacks=[cb],
        )

    # 7) Entrenar
    trainer.train()
    
    
    # 7b) Evaluación opcional en golden_set_test.csv (F1 vs reference_answer) del modelo finetuned
    # COMENTADO: Solo se evalúa vía Ollama+Agno (paso 10)
    # if getattr(args, "golden_eval_path", None):
    #     print("[eval][golden] Evaluando modelo finetuned (RLHF + QLoRA) después del entrenamiento...")
    #     evaluate_golden_set_f1(model=trainer.model,
    #         tokenizer=tok,
    #         golden_path=args.golden_eval_path,
    #         model_id=args.output_model_id,
    #         run_type="finetuned",
    #         method=args.method,
    #         results_dir="evaluation_results",
    #         max_samples=getattr(args, "golden_eval_max_samples", 0),
    #         max_new_tokens=getattr(args, "golden_eval_max_new_tokens", 128),
    #         batch_size=getattr(args, "golden_eval_batch_size", 4),
    #         use_chat_template=True,
    #         system_prompt=args.chat_system_prompt,
    #     )

    # 8) Directorio de salida (ya tenemos args.output_model_id resuelto arriba)
    out_id = args.output_model_id
    out_dir = os.path.join("./outputs", out_id)

    # 9) Guardar adaptador, merge 16-bit y GGUF/Ollama (pipeline 2-pasos HF->f16->q4_k_m)
    save_all(
        trainer,
        tok,
        out_dir,
        save_merged=not args.no_merge,
        save_gguf=bool(getattr(args, "save_gguf", True)),  # Por defecto True
        gguf_quant_method=str(getattr(args, "gguf_quant_method", "q4_k_m")),  # Default q4_k_m
        llama_cpp_convert_script=getattr(args, "llama_cpp_convert_script", None),
        modelfiles_dir=getattr(args, "modelfiles_dir", None),
        model_family=getattr(args, "model_family", None),
        ollama_bin=getattr(args, "ollama_bin", "ollama"),
        run_ollama_create=True,  # Por defecto registra en Ollama
    )
    
    # 9) (Opcional) Evaluación F1 del modelo base vía Ollama
    if args.base_to_ollama and getattr(args, "eval_base_ollama", False) and getattr(args, "golden_eval_path", None):
        print(f"[eval][base-ollama] Evaluando modelo base '{ollama_base_name}' vía Ollama + Agno...")
        evaluate_golden_set_f1_ollama(
            model_name=ollama_base_name,
            golden_path=args.golden_eval_path,
            model_id=ollama_base_name, 
            run_type="base-ollama",
            method="-",  # Modelo base: sin método de fine-tuning
            results_dir="evaluation_results",
            max_samples=getattr(args, "golden_eval_max_samples", 0),
            max_new_tokens=getattr(args, "golden_eval_max_new_tokens", 128),
            system_prompt=args.chat_system_prompt,
        )
    
    # 10) Evaluación F1 del modelo finetuned cuantizado ejecutado desde Ollama
    if getattr(args, "golden_eval_path", None):
        ft_ollama_name = args.output_model_id  # es el nombre que usa save_all/ollama create
        if ollama_model_exists(ft_ollama_name, ollama_bin=getattr(args, "ollama_bin", "ollama")):
            print(f"[eval][ft-ollama] Evaluando modelo finetuned cuantizado '{ft_ollama_name}' vía Ollama + Agno...")
            evaluate_golden_set_f1_ollama(
                model_name=ft_ollama_name,
                golden_path=args.golden_eval_path,
                model_id=args.output_model_id,
                run_type="ft-ollama",
                method=args.method,
                results_dir="evaluation_results",
                max_samples=getattr(args, "golden_eval_max_samples", 0),
                max_new_tokens=getattr(args, "golden_eval_max_new_tokens", 128),
                system_prompt=args.chat_system_prompt,
            )
        else:
            print(f"[eval][ft-ollama] Aviso: el modelo '{ft_ollama_name}' no aparece en 'ollama list'; se omite evaluación F1 vía Ollama.")

    print("[done] Entrenamiento finalizado.")

if __name__ == "__main__":
    main()
#
