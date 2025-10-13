#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pipeline de entrenamiento con QLoRA + Unsloth para DPO u ORPO
-----------------------------------------------------------------
- Carga un modelo base en 4-bit (QLoRA) vía Unsloth.
- Inserta adaptadores LoRA optimizados en las capas recomendadas.
- Prepara datasets de preferencias con columnas: prompt, chosen, rejected.
- Entrena con TRL usando DPOTrainer u ORPOTrainer (seleccionable por flag).
- Guarda el adaptador LoRA y (opcionalmente) guarda el modelo MERGEADO en 16-bit para despliegue (vLLM / TGI).

Requisitos (pip):
  pip install -U unsloth transformers datasets trl peft accelerate bitsandbytes

Ejemplos de uso:
  # DPO con dataset local en JSONL con columnas prompt, chosen, rejected
  accelerate launch pipeline_qlora_dpo_orpo_unsloth.py \
    --base_model unsloth/llama-3.2-3b-bnb-4bit \
    --method dpo \
    --data_path /ruta/dataset.jsonl \
    --output_dir ./outputs/llama32-3b-dpo \
    --epochs 2 --lr 2e-6 --beta 0.1

  # ORPO con dataset en Hugging Face Hub
  accelerate launch pipeline_qlora_dpo_orpo_unsloth.py \
    --base_model unsloth/llama-3.2-3b-bnb-4bit \
    --method orpo \
    --hf_dataset trl-lib/ultrafeedback_binarized \
    --split train \
    --output_dir ./outputs/llama32-3b-orpo \
    --epochs 2 --lr 8e-6

Notas:
- Por defecto se asume que el dataset ya contiene columnas explícitas: prompt, chosen, rejected.
  Si tus nombres difieren, usa --prompt_col, --chosen_col, --rejected_col.
- Para GPUs con bf16 soportado, se usará bf16; de lo contrario fp16.
- Para ahorrar VRAM se usa adamw_8bit + gradient checkpointing "unsloth".
"""

from __future__ import annotations
import os
import argparse
import json
from typing import Dict, List, Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict

from unsloth import FastLanguageModel, is_bfloat16_supported

# TRL
from trl import DPOTrainer, ORPOTrainer
from trl import DPOConfig, ORPOConfig  # opcional: podemos usar TrainingArguments también

# Transformers
from transformers import TrainingArguments


# -------------------------------
# Utilidades de dataset
# -------------------------------

def _read_json_or_jsonl(path: str) -> List[Dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            # aceptar {"data": [...]} etc.
            for k in ("data", "samples", "items"):
                if k in data and isinstance(data[k], list):
                    return data[k]
            raise ValueError("El JSON no es una lista. Provee JSONL o JSON con lista de ejemplos.")
        return data


def load_preference_dataset(
    data_path: Optional[str],
    hf_dataset: Optional[str],
    split: str,
    prompt_col: str,
    chosen_col: str,
    rejected_col: str,
) -> Dataset:
    """Devuelve un Dataset con columnas: prompt, chosen, rejected."""
    if hf_dataset:
        ds = load_dataset(hf_dataset, split=split)
        cols = ds.column_names
        # Mapea columnas si es necesario
        def _map_ex(example):
            prompt = example.get(prompt_col) or ""
            chosen = example.get(chosen_col)
            rejected = example.get(rejected_col)
            # Si prompt implícito está embebido en chosen/rejected, lo dejamos vacío.
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        ds = ds.map(_map_ex, remove_columns=[c for c in cols if c not in (prompt_col, chosen_col, rejected_col)])
        return ds
    else:
        if not data_path:
            raise ValueError("Debes pasar --data_path o --hf_dataset.")
        rows = _read_json_or_jsonl(data_path)
        # normalizar claves
        normed: List[Dict] = []
        for ex in rows:
            prompt = ex.get(prompt_col) or ""
            chosen = ex.get(chosen_col)
            rejected = ex.get(rejected_col)
            if chosen is None or rejected is None:
                raise ValueError("Cada ejemplo debe tener columnas 'chosen' y 'rejected'.")
            normed.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        return Dataset.from_list(normed)


# -------------------------------
# Entrenamiento
# -------------------------------

def build_model_and_tokenizer(
    base_model: str,
    max_seq_length: int,
    load_in_4bit: bool = True,
):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,  # unsloth decide bf16/fp16 según GPU más abajo en TrainingArguments
        load_in_4bit=load_in_4bit,
    )
    # Añadir LoRA en capas clave (atención + MLPs)
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    return model, tokenizer


def train_dpo(
    model, tokenizer, train_ds: Dataset, output_dir: str, epochs: int,
    lr: float, beta: float, per_device_bs: int, grad_accum: int,
    max_seq_length: int, max_prompt_length: int,
):
    # Podemos usar DPOConfig o TrainingArguments; seguimos práctica de Unsloth
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        num_train_epochs=epochs,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        report_to=["tensorboard"],
        run_name=os.path.basename(output_dir.rstrip("/")),
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,           # DPO ref-free (compatible con Unsloth)
        args=args,
        beta=beta,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
    )
    trainer.train()
    return trainer


def train_orpo(
    model, tokenizer, train_ds: Dataset, output_dir: str, epochs: int,
    lr: float, per_device_bs: int, grad_accum: int,
    max_seq_length: int, max_prompt_length: int,
):
    cfg = ORPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        num_train_epochs=epochs,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        optim="adamw_8bit",
        report_to=["tensorboard"],
        run_name=os.path.basename(output_dir.rstrip("/")),
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
    )

    trainer = ORPOTrainer(
        model=model,
        args=cfg,
        processing_class=tokenizer,
        train_dataset=train_ds,
    )
    trainer.train()
    return trainer


# -------------------------------
# Guardado
# -------------------------------

def save_all(
    trainer, tokenizer, output_dir: str,
    save_merged: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    # 1) Guardar adaptador LoRA (+ tokenizer)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # 2) (Opcional) Guardar modelo MERGEADO a 16-bit para vLLM/TGI
    if save_merged:
        try:
            # Método recomendado en docs de Unsloth para vLLM
            trainer.model.save_pretrained_merged(
                os.path.join(output_dir, "merged-16bit"), tokenizer,
                save_method="merged_16bit",
            )
        except Exception:
            # Fallback universal
            from unsloth import FastLanguageModel as _FLM
            merged = _FLM.merge_and_unload(trainer.model)
            merged.save_pretrained(os.path.join(output_dir, "merged-16bit-fallback"))
            tokenizer.save_pretrained(os.path.join(output_dir, "merged-16bit-fallback"))


# -------------------------------
# CLI
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="QLoRA + Unsloth: DPO/ORPO")
    p.add_argument("--base_model", type=str, default="unsloth/llama-3.2-3b-bnb-4bit")
    p.add_argument("--method", type=str, choices=["dpo", "orpo"], required=True)

    # dataset: o bien HF o un archivo local
    p.add_argument("--hf_dataset", type=str, default=None, help="ID en HuggingFace Hub (ej. trl-lib/ultrafeedback_binarized)")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--data_path", type=str, default=None, help="Ruta a JSON/JSONL con prompt/chosen/rejected")

    # columnas
    p.add_argument("--prompt_col", type=str, default="prompt")
    p.add_argument("--chosen_col", type=str, default="chosen")
    p.add_argument("--rejected_col", type=str, default="rejected")

    # training
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=None, help="LR. DPO recomendado ~1e-6 a 5e-6; ORPO ~8e-6")
    p.add_argument("--beta", type=float, default=0.1, help="beta para DPO")
    p.add_argument("--per_device_bs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1024)

    p.add_argument("--no_merge", action="store_true", help="No guardar el modelo mergeado 16-bit")
    return p.parse_args()


def main():
    args = parse_args()

    # Defaults de LR por método si no se fijó explícitamente
    if args.lr is None:
        if args.method == "dpo":
            args.lr = 2e-6
        else:  # orpo
            args.lr = 8e-6

    # 1) Cargar modelo + tokenizer con QLoRA
    model, tokenizer = build_model_and_tokenizer(
        base_model=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    # 2) Cargar dataset de preferencias
    train_ds = load_preference_dataset(
        data_path=args.data_path,
        hf_dataset=args.hf_dataset,
        split=args.split,
        prompt_col=args.prompt_col,
        chosen_col=args.chosen_col,
        rejected_col=args.rejected_col,
    )

    # 3) Entrenar
    if args.method == "dpo":
        trainer = train_dpo(
            model, tokenizer, train_ds, args.output_dir, args.epochs,
            args.lr, args.beta, args.per_device_bs, args.grad_accum,
            args.max_seq_length, args.max_prompt_length,
        )
    else:
        trainer = train_orpo(
            model, tokenizer, train_ds, args.output_dir, args.epochs,
            args.lr, args.per_device_bs, args.grad_accum,
            args.max_seq_length, args.max_prompt_length,
        )

    # 4) Guardar
    save_all(trainer, tokenizer, args.output_dir, save_merged=(not args.no_merge))


if __name__ == "__main__":
    main()
