from pathlib import Path
import pandas as pd
from jiwer import wer
import os 
import re
import unicodedata
import json
import argparse

def normalizar_es(texto: str) -> str:
    if not isinstance(texto, str):
        return ""

    # pasar a minúsculas
    texto = texto.lower()

    # quitar tildes: áéíóúü → aeiouu (ñ se mantiene)
    texto = unicodedata.normalize('NFD', texto)
    texto = "".join([c for c in texto if unicodedata.category(c) != 'Mn'])

    # quitar puntuación y símbolos
    texto = re.sub(r"[^a-z0-9ñ ]+", " ", texto)

    # colapsar espacios
    texto = re.sub(r"\s+", " ", texto).strip()

    return texto

def calcular_wer_con_normalizacion(
    datos_json: list,
    nombre_metodo: str,
    ruta_gold_csv: str,
    lista_ids
):
    """
    Calcula WER leyendo predicciones desde un JSON.
    
    Args:
        datos_json: Lista de diccionarios con los datos del JSON
        nombre_metodo: Nombre de la columna en el JSON (ej: "whisper_large-v3", "prediction_finetuned")
        ruta_gold_csv: Ruta al CSV con las referencias gold
        lista_ids: Lista de IDs a procesar
    """
    gold_df = pd.read_csv(ruta_gold_csv)

    # normalizar IDs y textos
    gold_df["id"] = gold_df["id"].astype(str)
    gold_df["context_text"] = gold_df["context_text"].astype(str)

    gold_map = dict(zip(gold_df["id"], gold_df["context_text"]))
    
    # Crear mapa de predicciones desde JSON
    predicciones_map = {}
    for item in datos_json:
        item_id = str(item.get("id", ""))
        prediccion = item.get(nombre_metodo)
        if prediccion is not None:
            predicciones_map[item_id] = str(prediccion).strip()

    refs = []
    hyps = []
    filas = []

    for _id in lista_ids:
        sid = str(_id)
        
        # Obtener predicción del JSON
        hyp = predicciones_map.get(sid)
        if hyp is None:
            print(f"[AVISO] Falta predicción para id={sid} en método {nombre_metodo}")
            continue

        ref = gold_map.get(sid)

        if ref is None:
            print(f"[AVISO] No hay gold para id={sid}")
            continue

        # Normalización español
        ref_norm = normalizar_es(ref)
        hyp_norm = normalizar_es(hyp)
        #  ref_norm = ref
        #  hyp_norm = hyp
        refs.append(ref_norm)
        hyps.append(hyp_norm)
        
        wer_ind = wer(ref_norm, hyp_norm)

        filas.append({
            "id": sid,
            "ref_original": ref,
            "hyp_original": hyp,
            "ref": ref_norm,
            "hyp": hyp_norm,
            "wer": wer_ind
        })

    wer_global = wer(refs, hyps) if refs else None
    df_detalle = pd.DataFrame(filas)
    return wer_global * 100, df_detalle


def agregar_gold_a_json(datos_json, ruta_gold_csv):
    """
    Añade la transcripción gold ('gold_transcription') al JSON de entrada
    emparejando por el campo 'id'.

    Args:
        datos_json (list): Lista de diccionarios del JSON original.
        ruta_gold_csv (str): Ruta al archivo CSV con columnas ['id', 'transcription'].

    Returns:
        list: Nueva lista de diccionarios con 'gold_transcription' añadido.
    """

    # Leer el CSV de gold
    gold_df = pd.read_csv(ruta_gold_csv)
    gold_df["id"] = gold_df["id"].astype(str)
    gold_map = dict(zip(gold_df["id"], gold_df["transcription"]))

    salida = []

    for item in datos_json:
        nuevo = item.copy()
        sid = str(item.get("id", ""))

        gold_text = gold_map.get(sid)

        # Añadir campo nuevo
        nuevo["gold_transcription"] = gold_text if gold_text is not None else ""

        salida.append(nuevo)

    return salida

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="WhisperX batch transcribe + align (solo LM y output por args).")
    parser.add_argument("--input_json", type=str, help="Ruta del JSON de entrada.")
    args = parser.parse_args()
    
    # Ruta al JSON de entrada con las predicciones
    ruta_json_entrada = args.input_json  # Ajusta esta ruta
    
    # Métodos a procesar (nombres de columnas en el JSON)
    metodos = {
        # "whisper_large-v3": "whisper_large-v3",
        "whisper_large-v3": "whisper_large-v3",
        "whisper_large-v3-turbo": "whisper_large-v3-turbo",
        "whisper_medium": "whisper_medium"
        # "whisper_tiny": "whisper_tiny",
        # "whisper_base": "whisper_base",
        # "whisper_small": "whisper_small"
    }
    
    csv_gold = "./dataset_full.csv"

    # Leer JSON de entrada
    with open(ruta_json_entrada, "r", encoding="utf-8") as f:
        datos_json = json.load(f)

    # Extraer IDs del JSON
    ids = [str(item.get("id", "")) for item in datos_json if item.get("id")]

    # Diccionario final agrupado POR ID
    resultados_por_id = {}

    # Inicializar estructura por ID
    for _id in ids:
        resultados_por_id[_id] = {}

    # Procesar cada método
    for nombre_modelo, nombre_columna in metodos.items():
        print(f"Procesando método: {nombre_modelo}")

        wer_total, df_por_utt = calcular_wer_con_normalizacion(
            datos_json, nombre_columna, csv_gold, ids
        )
        print(nombre_modelo, wer_total)
        df_por_utt["modelo"] = nombre_modelo
        df_por_utt["wer_global"] = wer_total

        # Convertimos cada fila en diccionario
        filas = df_por_utt.to_dict(orient="records")

        # Insertar cada fila dentro del ID correspondiente
        for fila in filas:
            _id = fila["id"]
            resultados_por_id[_id][nombre_modelo] = fila
