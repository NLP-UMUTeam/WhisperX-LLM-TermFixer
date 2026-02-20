
# WhisperX-LLM-TermFixer
A lightweight post-decoding framework to improve specific term recognition in Whisper-based system without fine-tuning or modifying the acoustic model or ASR like Whisper. This approach extends [WhisperX](https://github.com/m-bain/whisperX) framework with N-best decoding, lexicon-guided phonetic correction, and contextual re-scoring using a Large Language Model (LLM).

## Abstract
Large-scale end-to-end ASR models such as Whisper achieve strong general-domain performance but often fail to accurately recognize specialized terminology that is underrepresented in their training data, particularly in sensitive domains like medicine. In this paper, we propose a lightweight post-decoding framework to improve medical term recognition in Whisper-based systems without fine-tuning or modifying the acoustic model. Our approach extends WhisperX framework with N-best decoding, lexicon-guided phonetic correction, and contextual re-scoring using a Large Language Model. We evaluate the method on a controlled synthetic medical speech dataset containing 1,005 domain-specific terms. Results show consistent gains across Whisper model sizes, with term-level accuracy improvements exceeding 20 percentage points while also reducing Word Error Rate, demonstrating the effectiveness of post-decoding strategies for domain-specific ASR.

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
audio_path = "./test/audios_coqui/term_text_0.wav"
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
To evaluate medical term recognition under controlled conditions, we created a synthetic speech dataset containing 1,005 domain-specific medical and pharmaceutical terms (see `test` folder). 

Contextualized sentences were automatically generated using Gemini and converted to speech with Coqui TTS using multiple speakers. This setup ensures consistent pronunciation and controlled lexical coverage, enabling focused analysis of term recognition independently from acoustic noise or spontaneous speech variability.

### WER and Term-Level Accuracy Results

WER and term-level accuracy for Whisper models of different sizes using WhisperX under a baseline configuration and the proposed post-decoding approach combining beam search, phonetic filtering, and LLM-based contextual re-scoring. Results are reported for different LLMs (Gemma-2-9B, BioMistral-7B-Dare, and LLaMA-3.2-3B), together with absolute improvements in WER (ΔWER) and term accuracy (ΔAccuracy) relative to the baseline, as well as the average inference time per segment and total processing time.

#### Baseline

---

| Configuration | Model            | WER ↓ | % Accuracy ↑ | ΔWER (%) ↓ | ΔAccuracy (%) ↑ | Avg. Time / Segment (s) | Total Time (mm:ss) |
|---------------|------------------|-------|--------------|------------|------------------|--------------------------|--------------------|
| Baseline | Tiny           | 18.91 | 14.23 | - | - | 0.234 | 03:54 |
| Baseline | Base           | 13.66 | 19.90 | - | - | 0.235 | 03:56 |
| Baseline | Small          | 9.41  | 33.33 | - | - | 0.269 | 04:30 |
| Baseline | Medium         | 6.88  | 43.28 | - | - | 0.366 | 06:07 |
| Baseline | Large-v3-turbo | 6.50  | 44.88 | - | - | 0.282 | 04:43 |
| Baseline | Large-v3       | 5.24  | 55.62 | - | - | 0.426 | 07:07 |

---

#### LLM-Guided Post-Decoding

**Gemma-2-9B**

---

| Configuration | Model | WER ↓ | % Accuracy ↑ | ΔWER (%) ↓ | ΔAccuracy (%) ↑ | Avg. Time / Segment (s) | Total Time (mm:ss) |
|---------------|--------|-------|--------------|------------|------------------|--------------------------|--------------------|
| Gemma-2-9B | Tiny           | 16.85 | 35.52 | -2.06 | +21.29 | 0.390 | 06:32 |
| Gemma-2-9B | Base           | 11.08 | 49.85 | -2.58 | +29.95 | 0.410 | 06:52 |
| Gemma-2-9B | Small          | 7.22  | 62.29 | -2.19 | +28.96 | 0.544 | 09:07 |
| Gemma-2-9B | Medium         | 4.97  | 70.55 | -1.91 | +27.27 | 0.567 | 09:29 |
| Gemma-2-9B | Large-v3-turbo | 4.61  | 72.74 | -1.89 | +27.86 | 0.643 | 10:46 |
| Gemma-2-9B | Large-v3       | 3.61  | 78.81 | -1.63 | +23.19 | 0.818 | 13:42 |

---

**BioMistral-7B-dare**

---

| Configuration | Model | WER ↓ | % Accuracy ↑ | ΔWER (%) ↓ | ΔAccuracy (%) ↑ | Avg. Time / Segment (s) | Total Time (mm:ss) |
|---------------|--------|-------|--------------|------------|------------------|--------------------------|--------------------|
| BioMistral-7B-dare | Tiny           | 16.80 | 36.12 | -2.11 | +21.89 | 0.381 | 06:22 |
| BioMistral-7B-dare | Base           | 11.02 | 49.95 | -2.64 | +30.05 | 0.397 | 06:39 |
| BioMistral-7B-dare | Small          | 7.22  | 62.09 | -2.19 | +28.76 | 0.438 | 07:19 |
| BioMistral-7B-dare | Medium         | 4.97  | 70.55 | -1.91 | +27.27 | 0.540 | 09:02 |
| BioMistral-7B-dare | Large-v3-turbo | 4.62  | 72.54 | -1.88 | +27.66 | 0.444 | 07:26 |
| BioMistral-7B-dare | Large-v3       | 3.59  | 79.00 | -1.65 | +23.38 | 0.606 | 10:08 |

---

**LLaMA-3.2-3B**

---

| Configuration | Model | WER ↓ | % Accuracy ↑ | ΔWER (%) ↓ | ΔAccuracy (%) ↑ | Avg. Time / Segment (s) | Total Time (mm:ss) |
|---------------|--------|-------|--------------|------------|------------------|--------------------------|--------------------|
| LLaMA-3.2-3B | Tiny           | 16.81 | 36.12 | -2.10 | +21.89 | 0.366 | 06:07 |
| LLaMA-3.2-3B | Base           | 11.15 | 49.45 | -2.51 | +29.55 | 0.383 | 06:24 |
| LLaMA-3.2-3B | Small          | 7.24  | 61.89 | -2.17 | +28.56 | 0.422 | 07:04 |
| LLaMA-3.2-3B | Medium         | 5.00  | 70.15 | -1.88 | +26.87 | 0.535 | 08:57 |
| LLaMA-3.2-3B | Large-v3-turbo | 4.59  | 73.13 | -1.91 | +28.25 | 0.424 | 07:06 |
| LLaMA-3.2-3B | Large-v3       | 3.61  | 79.00 | -1.63 | +23.38 | 0.592 | 09:54 |
