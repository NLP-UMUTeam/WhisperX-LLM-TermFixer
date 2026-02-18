#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import gc
import argparse
from typing import List

import pandas as pd
import torch
from tqdm import tqdm
import whisperx
import ast

# ==========================
# CONFIGURACIÓN GLOBAL
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 16
MAX_LOADED_MODELS = 1

# Whisper versions
MODEL_MEDIUM = "medium"
MODEL_LARGE = "large-v3"
MODEL_TURBO = "large-v3-turbo"

MODEL_TINY = "tiny"
MODEL_BASE = "base"
MODEL_SMALL = "small"


# Dataset
DATASET_CSV = "./dataset_full.csv"
TERMINOS_MED = "./dataset_full.csv"

# Output (se puede sobreescribir por args)
OUTPUT_JSON = "/home/ronghao/interspeech/experiments_small_base_tiny/no_beam_no_lm.json"

# VAD (FIJO)
VAD_OPTIONS = {
    "model_fp": "/home/ronghao/vocali_genesis/dictado/test_hallucination/pytorch_model.bin",
    "vad_onset": 0.15,
    "vad_offset": 0.20,
}

# LM (CONTROLABLE POR ARGS)
USE_LM = True
LM_NAME = "/data/ronghao/LLM/google/gemma-2-9b/"
LM_THRESHOLD = 0.01
QUANTIZATION = "none"   # "none" | "4bit" | "8bit"
DEEP_LM = False

# ==========================
# UTILIDADES
# ==========================

def free_model(model):
    """Libera VRAM correctamente."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_term_list() -> List[str]:
    df = pd.read_csv(TERMINOS_MED)
    terms = df["term"].astype(str).str.lower().unique().tolist()
    return terms


def load_asr_model(model_name: str, term_list: List[str]):
    print(f"[ASR] Cargando modelo: {model_name}")
    return whisperx.load_model(
        model_name,
        DEVICE,
        compute_type=COMPUTE_TYPE,
        language="en",
        vad_options=VAD_OPTIONS,
        asr_options={"beam_size": 10},
        lexicon_terms=term_list,
        num_hypotheses=10,
        use_lm=USE_LM,
        deep_lm = DEEP_LM,
        lm_name=LM_NAME,
        lm_threshold=LM_THRESHOLD,
        quantization=QUANTIZATION,
        
    )


def load_aligner():
    print("[Align] Cargando modelo de alineación")
    return whisperx.load_align_model(language_code="en", device=DEVICE)


def transcribe_and_align(
    model,
    audio_path: str,
    align_model,
    metadata,
):
    start = time.time()

    result = model.transcribe(
        audio_path,
        task="transcribe",
        language="en",
        batch_size=BATCH_SIZE,
        chunk_size=30,
    )

    aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio_path,
        DEVICE,
        return_char_alignments=False,
    )

    text = " ".join(seg["text"] for seg in aligned["segments"])
    elapsed = time.time() - start
    return text, elapsed


# ==========================
# MAIN
# ==========================

def main():
    global USE_LM, LM_NAME, LM_THRESHOLD, QUANTIZATION, OUTPUT_JSON, DEEP_LM

    parser = argparse.ArgumentParser(description="WhisperX batch transcribe + align (solo LM y output por args).")

    # Output (args)
    parser.add_argument("--output_json", type=str, default=OUTPUT_JSON, help="Ruta del JSON de salida.")

    # LM ONLY (args)
    parser.add_argument("--use_lm", action="store_true", help="Activa deep LM analysis.")
    parser.add_argument("--deep_lm", action="store_true", help="Activa LM.")
    parser.add_argument("--no_lm", action="store_true", help="Desactiva LM (prioridad sobre --use_lm).")
    parser.add_argument("--lm_name", type=str, default=LM_NAME, help="Ruta/nombre del LM.")
    parser.add_argument("--lm_threshold", type=float, default=LM_THRESHOLD, help="Threshold del LM.")
    parser.add_argument("--quantization", type=str, choices=["none", "4bit", "8bit"], default=QUANTIZATION)

    args = parser.parse_args()

    # Aplicar args
    OUTPUT_JSON = args.output_json

    if args.no_lm:
        USE_LM = False
    elif args.use_lm:
        USE_LM = True  # si no se pasa nada, queda el default global

    LM_NAME = args.lm_name
    LM_THRESHOLD = args.lm_threshold
    QUANTIZATION = args.quantization
    DEEP_LM = args.deep_lm
    
    # Carga dataset y términos
    df = pd.read_csv(DATASET_CSV)
    df["id_path"] = df["id"].apply(lambda x: f"/home/ronghao/interspeech/audios_coqui/{x}.wav")
    term_list = load_term_list()

    # Alineador
    align_model, metadata = load_aligner()

    MODEL_CONFIGS = [
        ("whisper_tiny", MODEL_TINY),
        ("whisper_base", MODEL_BASE),
        ("whisper_small", MODEL_SMALL),
        ("whisper_medium", MODEL_MEDIUM),
        ("whisper_large-v3", MODEL_LARGE),
        ("whisper_large-v3-turbo", MODEL_TURBO),
    ]

    result_df = pd.DataFrame(columns=["id", "gold"])

    # Procesar modelos por tandas
    for i in range(0, len(MODEL_CONFIGS), MAX_LOADED_MODELS):
        batch = MODEL_CONFIGS[i:i + MAX_LOADED_MODELS]
        loaded_models = []

        try:
            # Cargar modelos
            for col_name, model_name in batch:
                model = load_asr_model(model_name, term_list)
                loaded_models.append((col_name, model))

            # Transcribir dataset
            for col_name, model in loaded_models:
                preds, times, ids, golds = [], [], [], []

                for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Transcribiendo {col_name}"):
                    try: 
                        text, t = transcribe_and_align(
                            model,
                            row["id_path"],
                            align_model,
                            metadata,
                        )
                        print(text)
                        preds.append(text)
                        ids.append(row["id"])
                        golds.append(row["context_text"])
                        times.append(t)
                    except:
                        pass

                result_df[col_name] = preds
                result_df[f"time_{col_name}"] = times
                result_df["id"] = ids
                result_df["gold"] = golds

        finally:
            # Liberar VRAM
            for _, model in loaded_models:
                free_model(model)

    os.makedirs(os.path.dirname(OUTPUT_JSON) or ".", exist_ok=True)
    result_df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

    print(f"\n[OK] Resultados guardados en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
