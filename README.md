
# WhisperX-LLM-TermFixer
A lightweight post-decoding framework to improve specific term recognition in Whisper-based system without fine-tuning or modifying the acoustic model or ASR like Whisper. This approach extends [WhisperX](https://github.com/m-bain/whisperX) framework with N-best decoding, lexicon-guided phonetic correction, and contextual re-scoring using a Large Language Model (LLM).

## Abstract
Large-scale ASR models such as Whisper perform well in general domains but struggle with specialized terminology, particularly in sensitive domains like medicine. We propose \textit{WhisperX-LLM-TermFixer}, a lightweight post-decoding framework that improves term recognition without retraining or modifying the acoustic model. The method combines $N$-best decoding, phonetic n-gram correction, and LLM-based contextual filtering. We evaluate our approach on a multilingual synthetic dataset covering English, Spanish, Catalan, and Basque, using the same set of medical terms across languages. 

## Method Overview

Given an audio segment:
1. **Whisper (faster-whisper) N-best decoding** generates *K* hypotheses.
2. **Lexicon re-ranking** selects the hypothesis with the best (term count, ASR score).
3. **Phonetic lexicon correction** proposes replacements over n-grams (up to `max_ngram`) using
   - text similarity + phonetic similarity (metaphone)
   - fast candidate retrieval via lexicon index (first letter + length bucket)
4. **LLM re-scoring (optional)** validates candidate replacements by comparing LM loss before/after:
   - accept only if improvement ≥ `lm_threshold`
   - optional `deep_lm`: searches best *prefix-span* replacement to avoid partial leftovers
5. **WhisperX alignment** aligns final segments to timestamps.



## Instalation
GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the  [CTranslate2 documentation](https://opennmt.net/CTranslate2/installation.html)

### 1 Create Python environment
```bash
conda create --name whisperx python=3.12
conda activate whisperx
```

### 2. Install the library

```bash
git clone https://github.com/NLP-UMUTeam/WhisperX-LLM-TermFixer.git
pip install -r requirements.txt
```

### 3. CUDA / cuDNN Library Path (Optional)
If you encounter runtime errors related to cuDNN, cuBLAS, or missing CUDA shared libraries (e.g. `libcudnn.so not found`), you may need to manually export the cuDNN library path from your conda environment:

```bash
export LD_LIBRARY_PATH=/home/XXXX/miniconda3/envs/whisperx/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```
Replace `whisperx` with your actual conda environment name.

To make this permanent, add the line to your `~/.bashrc`:

```bash
echo 'export LD_LIBRARY_PATH=/home/XXXX/miniconda3/envs/whisperx/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

Then reload:
```bash
source ~/.bashrc
```

## Usage Examples

```python
import pandas as pd
import whisperx
import torch
# 1) Device
device = "cuda" if torch.cuda.is_available() else "cpu"
# 2) Load term list (lexicon)
df_terms = pd.read_csv("./test/dataset/dataset_full.csv")
term_list = df_terms["term"].astype(str).str.lower().unique().tolist()
# 3) VAD options (example)
vad_options = {
    # "model_fp": "/path/to/pytorch_model.bin",
    "vad_onset": 0.15,
    "vad_offset": 0.20,
}
# 4) Load ASR model (WhisperX) + optional LM
model = whisperx.load_model(
    "large-v3-turbo",
    device,
    compute_type="float16",
    language="en",
    vad_options=vad_options,
    asr_options={"beam_size": 10},
    lexicon_terms=term_list,
    num_hypotheses=10,
    use_lm=True,
    deep_lm=True,
    # lm_name = meta-llama/Llama-3.2-3B
    # lm_name = google/gemma-2-9b
    lm_name="BioMistral/BioMistral-7B-DARE",
    lm_threshold=0.01,
    quantization="none",   # "none" | "4bit" | "8bit"
)
# 5) Load aligner
align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
# 6) Transcribe
audio_path = "./test/audios_english_tts/term_text_0.wav"
result = model.transcribe(
    audio_path,
    task="transcribe",
    language="en",
    batch_size=16,
    chunk_size=30,
)
# 7) Align segments and collect final text
aligned = whisperx.align(
    result["segments"],
    align_model,
    metadata,
    audio_path,
    device,
    return_char_alignments=False,
)
text = " ".join(seg["text"] for seg in aligned["segments"])
print(text)
```

### Audio Processing Note (reproducibility)

Keep **consistent audio preprocessing** across: 
- LM threshold tuning
- experiments / evaluation
- final inference 

Different resampling and preprocessing pipelines can change internal logits and affects LM acceptance decisions. 

### Configuration Summary

#### Term fix
- `lexicon_term`: list of domain terms (lowercased recommended).
- `num_hypotheses`: *K* hypotheses per segment (N-best). 
- Phonetic replacement:
    - max n-gram window (e.g. 7).
    - dynamic similarity thresholding (length-aware).
    - preserves punctuation and case 
    - optional accent restoration using lexicon map. 

- LLM Re-scoring (optional)   
    - `lm_name`: Identifier of the language model used for re-scoring. It can correspond to a local model path or a Hugging Face repository ID (e.g., `BioMistral/BioMistral-7B-DARE`, `meta-llama/Llama-3.2-3B` and `google/gemma-2-9b`). 
    - `use_lm`: enable/disable LLM scoring.
    - `deep_lm`: prefix-span search to avoid partial leftover tokens.
    - `lm_threshold`: minimum loss improvement to accept a replacement.
    - `quantization`: `none`, `4bit`, `8bit`.

#### Summary

| Parameter | Type | Default | Description | How to Use |
|------------|------|----------|-------------|-------------|
| `lexicon_terms` | `List[str]` | `None` | List of domain-specific terms used for lexicon-aware re-ranking and phonetic correction. | Pass a list of terms: `lexicon_terms=term_list`. Lowercased format is recommended. |
| `num_hypotheses` | `int` | `1` | Number of N-best hypotheses generated by Whisper per segment. Enables lexicon-based re-ranking across alternatives. | Increase (e.g., `10`) to allow more candidate hypotheses. Higher values increase computation time. |
| `use_lm` | `bool` | `False` | Enables contextual re-scoring using a causal Language Model. | Set `use_lm=True` to activate LM validation of phonetic replacements. |
| `lm_name` | `str` | `None` | Path or Hugging Face model ID of the LLM used for re-scoring. | Example: `"google/gemma-2-9b"` or `"/local/path/to/model"`. |
| `deep_lm` | `bool` | `False` | Enables prefix-span search during LM validation to avoid partial leftover tokens. | Use `deep_lm=True` for stricter contextual validation (recommended for multi-word terms). |
| `lm_threshold` | `float` | `0.01` | Minimum LM loss improvement required to accept a phonetic replacement. | Increase (e.g., `0.05`) for stricter corrections. Lower values allow more replacements. |
| `quantization` | `str` | `"none"` | Controls LLM quantization mode for memory efficiency. | Options: `"none"`, `"4bit"`, `"8bit"`. Use quantization for large models on limited GPU memory. |
---

#### Internal Phonetic Replacement Strategy

- **Max n-gram window:** up to 7 words.
- **Dynamic similarity thresholding:** length-aware filtering to reduce false positives.
- **Phonetic similarity:** combines text similarity (RapidFuzz) + Metaphone encoding.
- **Accent restoration:** restores canonical diacritics using lexicon mapping.
- **Case and punctuation preservation:** original formatting is preserved during replacement.

## Experiments
To evaluate medical term recognition under controlled conditions, we created a synthetic speech dataset containing 1,005 domain-specific medical and pharmaceutical terms (see `test` folder) in Spanish, English, Catalan and Basque. 

### WER and Term-Level Accuracy Results

WER and term-level accuracy for Whisper models of different sizes using WhisperX under a baseline configuration and the proposed post-decoding approach combining beam search, phonetic filtering, and LLM-based contextual re-scoring. Results are reported for best-performing LLM within each model family (Gemma, BioMistral, LLaMA, and Salamandra) for each
Whisper model size and language. Results are reported in terms of WER and term-level accuracy (%). 

#### English
---

| Whisper | Gemma WER | Gemma Acc | BioMistral WER | BioMistral Acc | LLaMA WER | LLaMA Acc | Salamandra WER | Salamandra Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 16.85 | 35.52 | 16.80 | 36.12 | 16.81 | 36.12 | **16.79** | **36.22** |
| Base | 11.08 | 49.85 | **11.02** | **49.95** | 11.05 | 49.95 | 11.04 | 49.95 |
| Small | 7.22 | 62.29 | 7.21 | 62.09 | 7.23 | 62.19 | **7.17** | **62.59** |
| Medium | 4.97 | 70.55 | 4.97 | 70.55 | 4.99 | 70.45 | **4.93** | **70.95** |
| Large-v3-turbo | 4.61 | 72.74 | 4.61 | 72.74 | 4.55 | 73.73 | **4.54** | **73.83** |
| Large-v3 | 3.61 | 78.81 | 3.59 | 79.00 | 3.61 | 79.00 | **3.55** | **79.20** |

#### Spanish
--- 

| Whisper | Gemma WER | Gemma Acc | BioMistral WER | BioMistral Acc | LLaMA WER | LLaMA Acc | Salamandra WER | Salamandra Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 14.57 | 67.06 | 14.54 | 67.06 | 14.58 | 66.87 | **14.63** | **67.16** |
| Base | 8.77 | 77.81 | **8.72** | **77.91** | 8.75 | 77.61 | 8.73 | 77.81 |
| Small | **4.46** | **85.97** | 4.52 | 85.47 | 4.46 | 85.87 | 4.51 | 85.67 |
| Medium | **2.50** | **90.95** | 2.50 | 90.95 | 2.50 | 90.85 | 2.51 | 90.95 |
| Large-v3-turbo | **1.85** | **93.63** | 1.86 | 93.23 | 1.87 | 93.33 | 1.87 | 93.23 |
| Large-v3 | 1.81 | 93.73 | 1.79 | 93.73 | **1.81** | **93.83** | **1.81** | **93.83** |

#### Catalan
--- 

| Whisper | Gemma WER | Gemma Acc | BioMistral WER | BioMistral Acc | LLaMA WER | LLaMA Acc | Salamandra WER | Salamandra Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 41.78 | 30.75 | 41.79 | 30.75 | 41.75 | 31.24 | **41.71** | **31.34** |
| Base | 27.91 | 39.30 | 28.01 | 38.91 | 27.94 | 39.20 | **27.78** | **39.80** |
| Small | 15.66 | 50.05 | 15.62 | 49.85 | 15.63 | 50.45 | **15.51** | **50.65** |
| Medium | 9.55 | 60.00 | 9.59 | 59.30 | **9.54** | **60.20** | 9.50 | 60.10 |
| Large-v3-turbo | 6.85 | 64.58 | 6.95 | 63.78 | 6.83 | 65.07 | **6.80** | **65.17** |
| Large-v3 | 6.28 | 70.05 | 6.30 | 70.05 | 6.30 | 70.15 | **6.22** | **70.35** |

#### Basque
--- 

| Whisper | Gemma WER | Gemma Acc | BioMistral WER | BioMistral Acc | LLaMA WER | LLaMA Acc | Salamandra WER | Salamandra Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tiny | 94.18 | 31.64 | 94.38 | 33.03 | 94.15 | 33.33 | **93.76** | **34.93** |
| Base | 87.72 | 36.62 | 87.80 | 39.50 | 87.63 | 40.00 | **87.32** | **41.89** |
| Small | 67.92 | 42.69 | 68.04 | 43.88 | 67.59 | 45.87 | **67.47** | **46.27** |
| Medium | 53.44 | 45.57 | 53.82 | 45.97 | 53.21 | 46.87 | **53.05** | **47.76** |
| Large-v3-turbo | 32.59 | 59.40 | 33.01 | 58.81 | 32.62 | 58.81 | **32.44** | **60.60** |
| Large-v3 | 34.21 | 54.53 | 34.47 | 55.82 | 34.12 | 55.52 | **33.95** | **56.02** |