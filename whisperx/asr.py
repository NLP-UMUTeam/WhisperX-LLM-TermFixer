import os
import warnings
from typing import List, NamedTuple, Optional, Union
from dataclasses import replace
import re

import ctranslate2
import faster_whisper
import numpy as np
import torch
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.transcribe import TranscriptionOptions, get_ctranslate2_storage
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .types import SingleSegment, TranscriptionResult
from .vad import VoiceActivitySegmentation, load_vad_model, merge_chunks
import unicodedata

from rapidfuzz import fuzz
import phonetics
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Mxfp4Config



def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        encoder_output=None,
        num_hypotheses: int = 5,
    ):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                
                num_hypotheses=max(1, int(num_hypotheses)),
                return_scores=True,
            )
        
        # Codigo de antes
        # tokens_batch = [x.sequences_ids[0] for x in result]
        
        tokens_batch = []
        scores_batch = []
        for x in result:
            hyps = x.sequences_ids[: max(1, int(num_hypotheses))]
            scs = x.scores[: len(hyps)] if getattr(x, "scores", None) is not None else [0.0] * len(hyps)
            tokens_batch.append(hyps)
            scores_batch.append(scs)

        # Antes 
        # def decode_batch(tokens: List[List[int]]) -> str:
        #     res = []
        #     for tk in tokens:
        #         res.append([token for token in tk if token < tokenizer.eot])
        #     # text_tokens = [token for token in tokens if token < self.eot]
        #     return tokenizer.tokenizer.decode_batch(res)

        
        def _decode_one(hyps_tokens: List[List[int]]) -> List[str]:
            # Decodifica N hipótesis de un sample (filtrando eot)
            res = []
            for tk in hyps_tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            return tokenizer.tokenizer.decode_batch(res)

        decoded_batch = [_decode_one(hyps) for hyps in tokens_batch] 
        
        # text = decode_batch(tokens_batch)
        
        # return text
        return decoded_batch, scores_batch

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
        self,
        model: WhisperModel,
        vad: VoiceActivitySegmentation,
        vad_params: dict,
        options: TranscriptionOptions,
        tokenizer: Optional[Tokenizer] = None,
        device: Union[int, str, "torch.device"] = -1,
        framework="pt",
        language: Optional[str] = None,
        suppress_numerals: bool = False,
    
        # News 
        lexicon_terms: Optional[List[str]] = None,
        num_hypotheses: int = 5,
        
        use_lm: bool = False,
        lm_name: Optional[str] = None,
        lm_threshold: float = 0.01,
        quantization: str = "none",   # "none" | "4bit" | "8bit"
        
        deep_lm: bool = False,

        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device
            
        #New
        self.lexicon_terms = [t.lower() for t in (lexicon_terms or [])]
        self.num_hypotheses = max(1, int(num_hypotheses))
        self._WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
        
        
        # --- Normalización para tildes / acentos ---
        def _normalize(s: str) -> str:
            # minúsculas + quitar tildes/acentos
            s = s.lower()
            nfkd = unicodedata.normalize("NFD", s)
            return "".join(ch for ch in nfkd if unicodedata.category(ch) != "Mn")

        self._normalize = _normalize

        # Mapa de forma normalizada -> forma canónica (con tildes)
        # Ojo: si hay colisiones, gana el último, que suele ser aceptable
        self._lexicon_norm_map = {}
        for term in self.lexicon_terms:
            nterm = self._normalize(term)
            self._lexicon_norm_map[nterm] = term

        self.use_lm = bool(use_lm)
        self.lm_name = lm_name
        self.lm_threshold = float(lm_threshold)
        self.quantization = (quantization or "none").lower().strip()
        self.deep_lm = bool(deep_lm)
        
        self._lexicon_norm_terms = []           # lista paralela a lexicon_terms
        self._lexicon_fast_index = {}           # (char, length) -> [idx, ...]

        for idx, term in enumerate(self.lexicon_terms):
            nt = self._normalize(term)          # ya sin tildes, minúsculas
            self._lexicon_norm_terms.append(nt)

            compact = re.sub(r"\s+", "", nt)    # quitamos espacios
            if not compact:
                continue
            key = (compact[0], len(compact))
            bucket = self._lexicon_fast_index.setdefault(key, [])
            bucket.append(idx)
            
        self.lm_tokenizer = None
        self.lm_model = None
        
        if self.use_lm:
            try:
                if not self.lm_name:
                    raise ValueError("use_lm=True pero lm_name es None/vacío")

                model_kwargs = dict(device_map="auto")

                lm_name_l = self.lm_name.lower()
                is_gpt_oss = "gpt-oss" in lm_name_l
                
                # Cuantización opcional
                if self.quantization == "4bit":
                    if is_gpt_oss:
                        print("Using Mxfp4Config for GPT-OSS")
                        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
                        model_kwargs["dtype"] = torch.bfloat16
                    else:   
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        model_kwargs["quantization_config"] = quantization_config

                elif self.quantization == "8bit":
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    model_kwargs["quantization_config"] = quantization_config

                elif self.quantization == "none":
                    pass
                else:
                    raise ValueError(f"quantization inválido: {self.quantization} (usa 'none', '4bit' o '8bit')")

                self.lm_tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
                self.lm_model = AutoModelForCausalLM.from_pretrained(
                    self.lm_name,
                    **model_kwargs,
                )
                self.lm_model.eval()

                # pad_token para modelos tipo GPT
                if self.lm_tokenizer.pad_token is None:
                    self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
                    self.lm_model.config.pad_token_id = self.lm_tokenizer.pad_token_id

                print(f"[LM] Modelo causal cargado para re-score: {self.lm_name} (quant={self.quantization})")
            except Exception as e:
                print(f"[LM] No se pudo cargar el modelo LM: {e}")
                self.lm_tokenizer = None
                self.lm_model = None
            
        
        super(Pipeline, self).__init__()
        self.vad_model = vad
        self._vad_params = vad_params

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}
    
    # New
    
    def _encode_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Codifica una lista de textos usando el encoder configurado.
        Soporta:
        - SentenceTransformer (método .encode)
        - Tupla (tokenizer, model) de HuggingFace
        - Callable que devuelva np.ndarray
        """
        if self.lexicon_encoder is None:
            return None

        enc = self.lexicon_encoder

        # 1) SentenceTransformer o similares (tienen .encode)
        if hasattr(enc, "encode"):
            # sentence-transformers: encode(texts, convert_to_numpy=True)
            return np.asarray(enc.encode(texts, convert_to_numpy=True))

        # 2) Tupla (tokenizer, model) HuggingFace
        if isinstance(enc, tuple) and len(enc) == 2:
            tokenizer, model = enc
            model_device = getattr(model, "device", self.device)

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(model_device)

            with torch.no_grad():
                outputs = model(**inputs)

            # Usamos mean-pooling sobre last_hidden_state
            last_hidden = outputs.last_hidden_state  # (batch, seq, dim)
            attn_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq, 1)
            summed = (last_hidden * attn_mask).sum(dim=1)
            counts = attn_mask.sum(dim=1).clamp(min=1e-9)
            embs = summed / counts

            return embs.cpu().numpy()

        # 3) Callable genérico: texts -> np.ndarray
        if callable(enc):
            arr = enc(texts)
            return np.asarray(arr)

        raise TypeError(
            "lexicon_encoder debe ser un SentenceTransformer, "
            "una tupla (tokenizer, model) de HuggingFace, "
            "o un callable(List[str]) -> np.ndarray"
        )


    def _count_terms(self, text: str) -> int:
        if not self.lexicon_terms:
            return 0

        # Normalizamos palabras del texto y lexicon
        words = {self._normalize(w) for w in self._WORD_RE.findall(text.lower())}

        count = 0
        for t in self.lexicon_terms:
            nt = self._normalize(t)
            if nt in words:
                count += 1
        return count
    
    def _copy_case(self, src: str, target: str) -> str:
        """Replica el estilo de mayúsculas de src en target."""
        if src.isupper():
            return target.upper()
        if src.istitle():
            return target.title()
        if src.islower():
            return target.lower()
        # mixto raro, devolvemos target tal cual
        return target

    def _fix_accents_with_lexicon(self, text: str) -> str:
        """Corrige tildes usando el lexicon (solo si hay coincidencia normalizada)."""
        if not self._lexicon_norm_map:
            return text

        # Separar en tokens conservando espacios/puntuación
        tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
        new_tokens = []

        for tok in tokens:
            if self._WORD_RE.fullmatch(tok):
                nt = self._normalize(tok)
                canonical = self._lexicon_norm_map.get(nt)
                if canonical is not None and tok != canonical:
                    # Ajustar mayúsculas/minúsculas
                    print(f"Reemplazo '{tok}' → '{canonical}'")
                    tok = self._copy_case(tok, canonical)
            new_tokens.append(tok)

        return "".join(new_tokens)

    def _phonetic_similarity(self, phrase: str, term: str) -> float:
        """
        Devuelve un score 0–100 combinando similitud de texto y fonética.
        No toca el texto original; solo normaliza internamente.
        """
        # Normalizamos (minúsculas, sin tildes)
        p_norm = self._normalize(phrase)
        t_norm = self._normalize(term)

        # Comparación textual
        score_text = fuzz.ratio(p_norm, t_norm)

        # Comparación fonética: quitamos espacios para que "para acer amor"
        # se parezca más a "paracetamol"
        p_comp = p_norm.replace(" ", "")
        t_comp = t_norm.replace(" ", "")

        meta_p = phonetics.metaphone(p_comp)
        meta_t = phonetics.metaphone(t_comp)

        score_meta = fuzz.ratio(meta_p, meta_t)

        # Combinamos (ajustable)
        return 0.6 * score_text + 0.4 * score_meta

    def _score_sentence_lm(self, sentence: str) -> float:
        """
        Devuelve la loss del modelo LM para una frase.
        Menor loss = frase más probable.
        """
        if not self.use_lm or self.lm_tokenizer is None or self.lm_model is None:
            return 0.0

        sent = sentence.strip()
        if not sent:
            return 0.0

        tokenized = self.lm_tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="longest",
        )
        
        try:
            emb_device = self.lm_model.get_input_embeddings().weight.device
        except Exception:
            emb_device = next(self.lm_model.parameters()).device

        inputs = {k: v.to(emb_device, non_blocking=True) for k, v in tokenized.items()}
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=emb_device)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = self.lm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return float(outputs.loss.item())


    def _apply_replace_span(self, words: List[str], start: int, end: int, term: str) -> List[str]:
        """Devuelve una nueva lista de palabras con [start:end] reemplazado por term."""
        # Reutiliza la misma lógica de puntuación que ya tienes en _phonetic_lexicon_replace
        boundary_punct = ",.;:?!¡¿"
        leading_punct_re = re.compile(rf"^[{re.escape(boundary_punct)}]+")
        trailing_punct_re = re.compile(rf"[{re.escape(boundary_punct)}]+$")

        def attach_boundary_punct(candidate: str, tokens: List[str]) -> str:
            if not tokens:
                return candidate
            prefix = ""
            suffix = ""
            first_token = tokens[0]
            last_token = tokens[-1]

            lead_match = leading_punct_re.match(first_token)
            if lead_match:
                prefix = lead_match.group(0)

            trail_match = trailing_punct_re.search(last_token)
            if trail_match:
                suffix = trail_match.group(0)

            if not prefix and not suffix:
                return candidate
            return f"{prefix}{candidate}{suffix}"

        term_with_punct = attach_boundary_punct(term, words[start:end])
        new_words = words.copy()
        new_words[start:end] = [term_with_punct]
        return new_words

    def _best_deep_lm_span(
        self,
        current_words: List[str],
        start: int,
        end: int,
        term: str,
    ):
        """
        Prueba SOLO prefijos: [start:start+1] ... [start:end]
        y devuelve el mejor (menor loss) que además pase reglas de dominio
        (evitar duplicaciones por restos a la derecha).
        """
        if not (self.use_lm and self.lm_tokenizer is not None and self.lm_model is not None):
            return None

        N = end - start
        if N <= 0:
            return None

        term_tokens = set(self._normalize(term).replace("/", " ").replace("-", " ").split())
        punct_strip = ".,;:?!¡¿\"'()[]{}"

        best = None  # (loss, s, e, candidate_words, candidate_sentence)

        for L in range(1, N + 1):
            s = start
            e = start + L

            # --- Regla dominio: si es parcial y deja un token del término justo a la derecha, lo saltamos ---
            if e < end:
                leftover_tok = current_words[e]
                leftover_norm = self._normalize(leftover_tok).strip(punct_strip)
                if leftover_norm and leftover_norm in term_tokens:
                    # Ej: "Agnecaste" -> term "agni casti fructus" y queda "fructus" a la derecha
                    continue

            cand_words = self._apply_replace_span(current_words, s, e, term)
            cand_sentence = " ".join(cand_words)
            loss = self._score_sentence_lm(cand_sentence)

            if best is None or loss < best[0]:
                best = (loss, s, e, cand_words, cand_sentence)

        return best



    def _phonetic_lexicon_replace(
        self,
        text: str,
        max_ngram: int = 3,
        threshold: float = 80.0,
    ) -> str:
        """
        Recorre el texto en n-grams y sustituye fragmentos que se parecen
        fonética/textualmente a términos del lexicon.
        Optimizado para lexicones grandes (~100k términos) usando un índice
        por primera letra + longitud aproximada.

        Imprime cada reemplazo encontrado.
        """
        if not self.lexicon_terms:
            return text

        words = text.split()
        if not words:
            return text

        sugerencias = []  # (inicio, fin, termino, score, frase_original)
        boundary_punct = ",.;:?!¡¿"
        leading_punct_re = re.compile(rf"^[{re.escape(boundary_punct)}]+")
        trailing_punct_re = re.compile(rf"[{re.escape(boundary_punct)}]+$")

        def attach_boundary_punct(candidate: str, tokens: List[str]) -> str:
            """Preserva puntuación pegada al primer o último token original."""
            if not tokens:
                return candidate
            prefix = ""
            suffix = ""
            first_token = tokens[0]
            last_token = tokens[-1]

            lead_match = leading_punct_re.match(first_token)
            if lead_match:
                prefix = lead_match.group(0)

            trail_match = trailing_punct_re.search(last_token)
            if trail_match:
                suffix = trail_match.group(0)

            if not prefix and not suffix:
                return candidate
            return f"{prefix}{candidate}{suffix}"

        # Parámetros del umbral dinámico
        max_thr = float(threshold)           # para frases muy cortas
        min_thr = max(max_thr - 15.0, 80.0)  # no bajar de 80
        L_min = 5
        L_max = 12
        
        max_ngram = min(max_ngram, len(words))
        
        def dynamic_required(L: int) -> float:
            """Devuelve el score mínimo requerido en función de la longitud L."""
            if L <= L_min:
                return max_thr
            if L >= L_max:
                return min_thr
            # interpolación lineal
            return max_thr - (max_thr - min_thr) * (L - L_min) / (L_max - L_min)

        for n in range(1, max_ngram + 1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])

                # Normalizamos la frase (sin tildes, minúsculas)
                p_norm = self._normalize(phrase)
                p_compact = re.sub(r"\s+", "", p_norm)
                if not p_compact:
                    continue
                    
                L = len(p_compact)

                # Si es tremendamente corto, ni lo intentamos
                if L < 4:
                    continue

                # Si la frase normalizada YA es un término en el lexicón,
                # dejamos que lo maneje _fix_accents_with_lexicon
                if p_norm in self._lexicon_norm_map:
                    continue

                # Buscamos sólo en buckets con misma primera letra y longitudes cercanas
                first = p_compact[0]
                candidate_indices = []
                for delta in (-2, -1, 0, 1, 2):  # longitud +/- 2 chars
                    L_bucket = L + delta
                    if L_bucket < 1:
                        continue
                    key = (first, L_bucket)
                    bucket = self._lexicon_fast_index.get(key)
                    if bucket:
                        candidate_indices.extend(bucket)

                if not candidate_indices:
                    continue

                best_score = 0.0
                best_term = None

                for idx in set(candidate_indices):
                    term = self.lexicon_terms[idx]
                    score = self._phonetic_similarity(phrase, term)
                    if score > best_score:
                        best_score = score
                        best_term = term

                if best_term is None:
                    continue
                
                # --- UMBRAL DINÁMICO EN FUNCIÓN DE L ---
                base_thr = 87.0
                L_ref = 12       # longitud "media" de referencia
                alpha = 2.5      # ajuste por carácter
            
                required = threshold + alpha * (L_ref - L)
                # clamp: no bajar demasiado ni subir demasiado
                if required < 87.0:
                    required = 87.0
                if required > 96.0:
                    required = 96.0

                if best_score < required:
                    # Demasiado poco parecido para esta longitud
                    continue
                term_norm = self._normalize(best_term)
                phrase_norm = p_norm  # ya calculado antes
                phrase_tokens = phrase_norm.split()

                if len(phrase_tokens) > 1 and term_norm in phrase_tokens:
                    # Ej: 'radiografia de' -> 'radiografia'  ==> saltamos
                    continue
                
                if term_norm == phrase_norm:
                    continue


                sugerencias.append((i, i + n, best_term, best_score, phrase, L))
            

        if not sugerencias:
            return text

        # Ordenamos por score descendente y evitamos solapes
        # sugerencias.sort(
        #     key=lambda x: ((x[1] - x[0]), x[3]),
        #     reverse=True,
        # )
        sugerencias.sort(key=lambda x: x[0], reverse=True)
        print(sugerencias)
        ocupadas = set()

        current_words = words
        current_sentence = " ".join(current_words)
        current_loss = self._score_sentence_lm(current_sentence)
        use_lm = self.use_lm and (self.lm_tokenizer is not None) and (self.lm_model is not None)

        for start, end, term, score, phrase, L in sugerencias:
            idxs = set(range(start, end))
            if any(idx in ocupadas for idx in idxs):
                continue
            
            new_words = self._apply_replace_span(current_words, start, end, term)
            candidate_sentence = " ".join(new_words)

            chosen_start, chosen_end = start, end
            loss_cand = None

            if use_lm:
                if self.deep_lm:
                    best = self._best_deep_lm_span(current_words, start, end, term)
                    if best is None:
                        continue

                    loss_best, s_best, e_best, words_best, sent_best = best
                    improvement = current_loss - loss_best
                    if improvement < self.lm_threshold:
                        continue
                    
                    # Aceptamos el mejor prefijo
                    loss_cand = loss_best
                    chosen_start, chosen_end = s_best, e_best
                    new_words = words_best
                    candidate_sentence = sent_best

                    print(
                        f"[LM deep-seq] Acepto '{' '.join(current_words[chosen_start:chosen_end])}' -> '{term}' "
                        f"(loss_old={current_loss:.4f}, loss_new={loss_cand:.4f}, Δ={improvement:.4f}) "
                        f"span={chosen_start}:{chosen_end}"
                    )
                else:
                    # comportamiento anterior: span completo
                    loss_cand = self._score_sentence_lm(candidate_sentence)
                    improvement = current_loss - loss_cand
                    if improvement < self.lm_threshold:
                        continue
                    current_loss = loss_cand

            # Recalcular idxs/ocupación con el span finalmente elegido
            idxs_final = set(range(chosen_start, chosen_end))
            if any(idx in ocupadas for idx in idxs_final):
                continue

            # info debug 
            replaced_src = " ".join(current_words[chosen_start:chosen_end])
            print(
                f"[PhoneticReplace] '{replaced_src}'  →  '{term}'   "
                f"(score={score:.2f}, palabras={chosen_start}:{chosen_end}, L={L})"
            )

            current_words = new_words
            ocupadas.update(idxs_final)
        
        final_text = " ".join(current_words)
        print("[FINAL]", final_text)
        return final_text
    

    def _forward(self, model_inputs):
        
        # New
        decoded_batch, scores_batch = self.model.generate_segment_batched(
            model_inputs['inputs'], self.tokenizer, self.options, num_hypotheses=self.num_hypotheses
        )
                # decoded_batch: List[List[str]] (batch x K); scores_batch: List[List[float]] (batch x K)
        chosen = []
        for texts, scores in zip(decoded_batch, scores_batch):
            # print(texts, scores)
            if self.num_hypotheses == 1 or not self.lexicon_terms:
                chosen.append(texts[0] if texts else "")
                continue
            # Reranking: max por (term_count, score)
            counts = [self._count_terms(t) for t in texts]
            best_idx = max(range(len(texts)), key=lambda i: (counts[i], scores[i] if i < len(scores) else 0.0))
            chosen.append(texts[best_idx])

        
        chosen_phonetic = [
            self._phonetic_lexicon_replace(t, max_ngram=7, threshold=85.0)
            for t in chosen
        ]
        return {'text': chosen_phonetic}
    

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self,
        inputs,
        num_workers: int,
        batch_size: int,
        preprocess_params: dict,
        forward_params: dict,
        postprocess_params: dict,
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers=0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size=30,
        print_progress=False,
        combined_progress=False,
        verbose=False,
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        vad_segments = self.vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self._vad_params["vad_onset"],
            offset=self._vad_params["vad_offset"],
        )
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = Tokenizer(
                    self.model.hf_tokenizer,
                    self.model.model.is_multilingual,
                    task=task,
                    language=language,
                )

        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            print(f"Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = replace(self.options, suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            if verbose:
                print(f"Transcript: [{round(vad_segments[idx]['start'], 3)} --> {round(vad_segments[idx]['end'], 3)}] {text}")
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = replace(self.options, suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language}

    def detect_language(self, audio: np.ndarray) -> str:
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return language


def load_model(
    whisper_arch: str,
    device: str,
    device_index=0,
    compute_type="float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_model: Optional[VoiceActivitySegmentation] = None,
    vad_options: Optional[dict] = None,
    model: Optional[WhisperModel] = None,
    task="transcribe",
    download_root: Optional[str] = None,
    local_files_only=False,
    threads=4,
    # New 
    lexicon_terms: Optional[List[str]] = None,
    num_hypotheses: int = 1,
    
    use_lm: bool = False,
    lm_name: Optional[str] = None,
    lm_threshold: float = 0.01,
    quantization: str = "none",
    
    deep_lm: bool = False,

    
    
) -> FasterWhisperPipeline:
    """Load a Whisper model for inference.
    Args:
        whisper_arch - The name of the Whisper model to load.
        device - The device to load the model on.
        compute_type - The compute type to use for the model.
        options - A dictionary of options to use for the model.
        language - The language of the model. (use English for now)
        model - The WhisperModel instance to use.
        download_root - The root directory to download the model to.
        local_files_only - If `True`, avoid downloading the file and return the path to the local cached file if it exists.
        threads - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    """
    

    if whisper_arch.endswith(".en"):
        language = "en"

    model = model or WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type=compute_type,
                         download_root=download_root,
                         local_files_only=local_files_only,
                         cpu_threads=threads)
    if language is not None:
        tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'“¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：”)]}、",
        "multilingual": model.model.is_multilingual,
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": None,
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = TranscriptionOptions(**default_asr_options)

    default_vad_options = {
        "vad_onset": 0.500,
        "vad_offset": 0.363
    }

    if vad_options is not None:
        default_vad_options.update(vad_options)

    if vad_model is not None:
        vad_model = vad_model
    else:
        vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)

    return FasterWhisperPipeline(
        model=model,
        vad=vad_model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
        vad_params=default_vad_options,
        
        lexicon_terms=lexicon_terms,
        num_hypotheses=num_hypotheses,

        use_lm=use_lm, 
        lm_name=lm_name,
        lm_threshold=lm_threshold,
        quantization=quantization,
        
        deep_lm=deep_lm,
        
    )