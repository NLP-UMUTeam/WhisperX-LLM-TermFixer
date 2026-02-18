import json 
import re
import os
import glob
import unicodedata
import string
import pandas as pd 
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
    

def check_medical_termino_text(text, entidades):
    """
    Verifica si los términos médicos están presentes en el texto.
    
    Args:
        text: Texto a verificar (str)
        entidades: Lista de términos médicos a buscar (list)
    
    Returns:
        dict con los resultados del chequeo:
            - Para cada entidad: 1 si está presente, 0 si no
            - "total_correcto": número total de términos encontrados
    """
    if not isinstance(text, str):
        text = ""
        
    text = normalizar_es(text)
    
    entidades = [normalizar_es(entidad) for entidad in entidades]
    result_dict = {}
    count = 0
    for entidad in entidades:
        # Regex para buscar la entidad como frase completa, ignorando tildes y puntuación
        pattern = r'(?<!\w)' + re.escape(entidad) + r'(?!\w)'
        if re.search(pattern, text):
            result_dict[entidad] = 1
            count += 1
        else:
            result_dict[entidad] = 0
            
    result_dict["total_correcto"] = count
    
    return result_dict


def _missed_terms(result_dict: dict) -> list:
    """
    Devuelve la lista de términos NO encontrados (valor 0) en un dict de check_medical_termino_text.
    El dict viene con claves de término y 'total_correcto'.
    """
    if not isinstance(result_dict, dict):
        return []
    return [k for k, v in result_dict.items() if k != "total_correcto" and v == 0]


def terminology_check_single(prediction_json, terminos: list = None, terminos_map: dict = None, output_file: str = None):
    """
    Verifica la presencia de términos médicos en las predicciones de uno o más objetos JSON.
    
    Args:
        prediction_json: Puede ser:
            - str: Ruta a un archivo .json
            - dict: Diccionario con las predicciones
            - list: Lista de diccionarios con predicciones
        terminos: Lista de términos médicos esperados a verificar (list[str])
                 Solo se usa si terminos_map no se proporciona
        terminos_map: Diccionario con mapeo id -> lista de términos {id: [termino1, termino2, ...]}
                     Si se proporciona, se usará el id del prediction_json para obtener los términos
        output_file: (opcional) Ruta donde guardar los resultados en formato JSON. Si se proporciona,
                     los resultados se guardarán automáticamente en el archivo especificado.
    
    Returns:
        Si es un solo diccionario:
            dict con los resultados del chequeo individual
        
        Si es una lista:
            dict con:
                - "results": lista de dicts con resultados individuales
                - "summary": dict con summary global por columna, cada columna contiene:
                    - "total_files": número total de archivos
                    - "files_processed": número de archivos procesados
                    - "total_terms_expected": total de términos esperados
                    - "total_hits": total de hits
                    - "total_missed": total de términos no encontrados
                    - "global_percentage": porcentaje global
                    - "average_percentage": promedio de porcentajes
    """
    if terminos_map is None and terminos is None:
        raise ValueError("Debe proporcionarse terminos o terminos_map")
    
    # Si es un string, asumimos que es una ruta a un archivo JSON
    if isinstance(prediction_json, str):
        if not os.path.exists(prediction_json):
            raise ValueError(f"El archivo {prediction_json} no existe")
        try:
            with open(prediction_json, "r", encoding="utf-8") as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear el archivo JSON {prediction_json}: {e}")
        except Exception as e:
            raise ValueError(f"Error al leer el archivo {prediction_json}: {e}")
    else:
        json_data = prediction_json
    
    # Procesar según el tipo de datos
    if isinstance(json_data, list):
        # Si es una lista, procesar todos los elementos
        if len(json_data) == 0:
            raise ValueError("El JSON contiene una lista vacía")
        
        results = []
        # Primero detectar todas las columnas disponibles (usando el primer elemento)
        first_item = json_data[0]
        if not isinstance(first_item, dict):
            raise ValueError(f"Todos los elementos de la lista deben ser diccionarios. Encontrado: {type(first_item)}")
        
        known_columns = ["prediction_finetuned", "whisper_large-v3", "whisper_large-v3-turbo"]
        all_column_names = set()
        
        # Recopilar todas las columnas de todos los elementos
        for item in json_data:
            if not isinstance(item, dict):
                continue
            excluded_keys = {"id"} | {k for k in item.keys() if k.startswith("time_")}
            for col in known_columns:
                if col in item:
                    all_column_names.add(col)
            for key, value in item.items():
                if key not in excluded_keys and key not in all_column_names:
                    if isinstance(value, str) and value.strip():
                        all_column_names.add(key)
        
        all_column_names = sorted(list(all_column_names))
        
        if not all_column_names:
            raise ValueError("No se encontraron columnas de predicción en el JSON")
        
        # Inicializar acumuladores por columna para el summary global
        column_stats = {col: {"total_hits": 0, "total_terms_expected": 0, "percentages": []} 
                       for col in all_column_names}
        
        # Procesar cada elemento
        for item in json_data:
            if not isinstance(item, dict):
                continue
            
            file_id = item.get("id", "")
            if terminos_map is not None:
                terminos_for_item = terminos_map.get(file_id, [])
            else:
                terminos_for_item = terminos if terminos else []
            
            result = {
                "id": file_id,
                "terminos_esperados": terminos_for_item,
                "n_terminos_esperados": len(terminos_for_item)
            }
            
            # Verificar cada columna que exista en este item
            item_summary = {}
            for column_name in all_column_names:
                text = item.get(column_name, "")
                if not isinstance(text, str):
                    text = ""
                
                check_result = check_medical_termino_text(text, terminos_for_item)
                result[f"{column_name}_medical_termino"] = check_result
                
                missed = _missed_terms(check_result)
                result[f"{column_name}_missed_terms"] = missed
                hits = check_result.get("total_correcto", 0)
                result[f"{column_name}_hits"] = hits
                result[f"{column_name}_text"] = text
                
                # Calcular summary por item
                total_terms = len(terminos_for_item)
                percentage = (100 * hits / total_terms) if total_terms > 0 else 0.0
                item_summary[column_name] = {
                    "hits": hits,
                    "missed": total_terms - hits,
                    "percentage": round(percentage, 2)
                }
                
                # Acumular para summary global
                column_stats[column_name]["total_hits"] += hits
                column_stats[column_name]["total_terms_expected"] += total_terms
                column_stats[column_name]["percentages"].append(percentage)
            
            result["summary"] = item_summary
            result["failed"] = any(
                result[f"{col}_hits"] < len(terminos_for_item)
                for col in all_column_names
                if len(terminos_for_item) > 0
            )
            results.append(result)
        
        # Calcular summary global por columna
        global_summary = {}
        total_files = len(results)
        
        for column_name in all_column_names:
            stats = column_stats[column_name]
            total_hits = stats["total_hits"]
            total_terms_expected = stats["total_terms_expected"]
            global_percentage = (100 * total_hits / total_terms_expected) if total_terms_expected > 0 else 0.0
            average_percentage = sum(stats["percentages"]) / len(stats["percentages"]) if stats["percentages"] else 0.0
            
            global_summary[column_name] = {
                "total_files": total_files,
                "files_processed": len([r for r in results if f"{column_name}_hits" in r]),
                "total_terms_expected": total_terms_expected,
                "total_hits": total_hits,
                "total_missed": total_terms_expected - total_hits,
                "global_percentage": round(global_percentage, 2),
                "average_percentage": round(average_percentage, 2)
            }
        
        final_result = {
            "results": results,
            "summary": global_summary
        }
        
        # Guardar en archivo si se especificó
        if output_file:
            try:
                # Crear directorio si no existe
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(final_result, f, ensure_ascii=False, indent=2)
                print(f"Resultados guardados en: {output_file}")
            except Exception as e:
                print(f"Error al guardar el archivo {output_file}: {e}")
        
        return final_result
        
    elif isinstance(json_data, dict):
        # Si es un diccionario, procesar solo ese elemento
        file_id = json_data.get("id", "")
        if terminos_map is not None:
            terminos_for_item = terminos_map.get(file_id, [])
        else:
            terminos_for_item = terminos if terminos else []
        
        # Detectar columnas
        known_columns = ["firework", "prediction_finetuned", "whisper_large-v3", "whisper_large-v3-turbo"]
        column_names = []
        
        for col in known_columns:
            if col in json_data:
                column_names.append(col)
        
        excluded_keys = {"id"} | {k for k in json_data.keys() if k.startswith("time_")}
        for key, value in json_data.items():
            if key not in excluded_keys and key not in column_names:
                if isinstance(value, str) and value.strip():
                    column_names.append(key)
        
        if not column_names:
            raise ValueError("No se encontraron columnas de predicción en el JSON")
        
        result = {
            "id": file_id,
            "terminos_esperados": terminos_for_item,
            "n_terminos_esperados": len(terminos_for_item)
        }
        
        summary = {}
        total_terms = len(terminos_for_item)
        
        for column_name in column_names:
            text = json_data.get(column_name, "")
            if not isinstance(text, str):
                text = ""
            
            check_result = check_medical_termino_text(text, terminos_for_item)
            result[f"{column_name}_medical_termino"] = check_result
            
            missed = _missed_terms(check_result)
            result[f"{column_name}_missed_terms"] = missed
            hits = check_result.get("total_correcto", 0)
            result[f"{column_name}_hits"] = hits
            result[f"{column_name}_text"] = text
            
            percentage = (100 * hits / total_terms) if total_terms > 0 else 0.0
            summary[column_name] = {
                "hits": hits,
                "missed": total_terms - hits,
                "percentage": round(percentage, 2)
            }
        
        result["summary"] = summary
        result["failed"] = any(
            result[f"{col}_hits"] < total_terms
            for col in column_names
            if total_terms > 0
        )
        
        # Guardar en archivo si se especificó
        if output_file:
            try:
                # Crear directorio si no existe
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"Resultados guardados en: {output_file}")
            except Exception as e:
                print(f"Error al guardar el archivo {output_file}: {e}")
        
        return result
    else:
        raise ValueError(f"El JSON debe contener un diccionario o una lista de diccionarios. Tipo recibido: {type(json_data)}")


def terminology_check_folder(folder_path: str, terminos_map: dict = None, terminos_global: list = None, output_file: str = None) -> dict:
    """
    Verifica términos médicos en todos los archivos {id}.txt de una carpeta.
    
    Args:
        folder_path: Ruta a la carpeta que contiene los archivos {id}.txt
        terminos_map: Diccionario con mapeo id -> lista de términos {id: [termino1, termino2, ...]}
                     Si se proporciona, cada archivo usará sus términos específicos
        terminos_global: Lista única de términos para todos los archivos
                        Solo se usa si terminos_map no se proporciona
        output_file: (opcional) Ruta donde guardar los resultados en formato JSON. Si se proporciona,
                     los resultados se guardarán automáticamente en el archivo especificado.
    
    Returns:
        dict con los resultados:
            - "results": lista de dicts, uno por archivo procesado
            - "summary": resumen global con estadísticas
    """
    if terminos_map is None and terminos_global is None:
        raise ValueError("Debe proporcionarse terminos_map o terminos_global")
    
    if not os.path.exists(folder_path):
        raise ValueError(f"La carpeta {folder_path} no existe")
    
    if not os.path.isdir(folder_path):
        raise ValueError(f"{folder_path} no es una carpeta")
    
    # Buscar todos los archivos .txt en la carpeta
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    
    results = []
    total_files = len(txt_files)
    total_hits = 0
    total_terms_expected = 0
    
    for txt_file in txt_files:
        # Extraer el id del nombre del archivo (sin extensión)
        file_id = os.path.splitext(os.path.basename(txt_file))[0]
        
        # Determinar qué términos usar para este archivo
        if terminos_map is not None:
            terminos = terminos_map.get(file_id, [])
        else:
            terminos = terminos_global if terminos_global else []
        
        # Leer el contenido del archivo
        try:
            with open(txt_file, "r", encoding="utf-8") as f:
                transcription = f.read().strip()
        except Exception as e:
            print(f"Error leyendo {txt_file}: {e}")
            continue
        
        # Verificar términos
        check_result = check_medical_termino_text(transcription, terminos)
        hits = check_result.get("total_correcto", 0)
        n_terminos = len(terminos)
        percentage = (100 * hits / n_terminos) if n_terminos > 0 else 0.0
        
        # Guardar resultado
        result_item = {
            "id": file_id,
            "file_path": txt_file,
            "terminos_esperados": terminos,
            "n_terminos_esperados": n_terminos,
            "hits": hits,
            "missed": n_terminos - hits,
            "percentage": round(percentage, 2),
            "missed_terms": _missed_terms(check_result),
            "transcription": transcription
        }
        results.append(result_item)
        
        # Acumular para resumen global
        total_hits += hits
        total_terms_expected += n_terminos
    
    # Calcular resumen global
    global_percentage = (100 * total_hits / total_terms_expected) if total_terms_expected > 0 else 0.0
    
    summary = {
        "total_files": total_files,
        "files_processed": len(results),
        "total_terms_expected": total_terms_expected,
        "total_hits": total_hits,
        "total_missed": total_terms_expected - total_hits,
        "global_percentage": round(global_percentage, 2),
        "average_percentage": round(sum(r["percentage"] for r in results) / len(results), 2) if results else 0.0
    }
    
    final_result = {
        "results": results,
        "summary": summary
    }
    
    # Guardar en archivo si se especificó
    if output_file:
        try:
            # Crear directorio si no existe
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            print(f"Resultados guardados en: {output_file}")
        except Exception as e:
            print(f"Error al guardar el archivo {output_file}: {e}")
    
    return final_result


def create_terminology_map_from_csv(csv_path: str) -> dict:
    """
    Crea un mapeo de terminología desde un CSV con columnas:
        - id
        - termino

    Cada id tiene UN solo término asociado.

    Args:
        csv_path: Ruta al archivo CSV

    Returns:
        dict con el mapeo {id: termino}
    """
    if not os.path.exists(csv_path):
        raise ValueError(f"El archivo CSV {csv_path} no existe")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error leyendo el CSV {csv_path}: {e}")

    required_cols = {"id", "term"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"El CSV debe contener las columnas {required_cols}, "
            f"pero tiene {set(df.columns)}"
        )

    terminology_map = {}

    for _, row in df.iterrows():
        file_id = str(row["id"]).strip()
        termino = str(row["term"]).strip()

        if not file_id or not termino or termino.lower() == "nan":
            continue

        # Si el id se repite, avisa y sobreescribe (o puedes cambiar el comportamiento)
        if file_id in terminology_map:
            print(f"Advertencia: id duplicado '{file_id}', se sobreescribe el término")

        terminology_map[file_id] = [termino]

    return terminology_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperX batch transcribe + align (solo LM y output por args).")
    parser.add_argument("--input_json", type=str, help="Ruta del JSON de entrada.")
    parser.add_argument("--output_json", type=str, help="Ruta del JSON de entrada.")
    args = parser.parse_args()
    
    
    terminos_map = create_terminology_map_from_csv(
        "dataset_full.csv"
    )
       
    
    resultado = terminology_check_single(args.input_json, 
                                         terminos_map=terminos_map,
                                         output_file=args.output_json)
        