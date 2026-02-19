
# WhisperX-LexiCorrect-LM
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

### 2. Install PyTorch, e.g. for Linux and Windows CUDA11.8:

`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

```bash
git clone https://github.com/NLP-UMUTeam/interspeech-2026-LexiCorrect-ASR.git
cd whisperX
pip install -e .
```

## Usage Examples

```python
import pandas as pd
import whisperx
import torch
# 1) Device
device = "cuda" if torch.cuda.is_available() else "cpu"
# 2) Load term list (lexicon)
df_terms = pd.read_csv("./dataset_full.csv")
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
    deep_lm=False,
    lm_name="/path/to/lm/",
    lm_threshold=0.01,
    quantization="none",   # "none" | "4bit" | "8bit"
)
# 5) Load aligner
align_model, metadata = whisperx.load_align_model(language_code="en", device=device)
# 6) Transcribe
audio_path = "examples/sample.wav"
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

```
### Audio Processing Note (reproducibility)

Keep **consistent audio preprocessing** across: 
- LM threshold tuning
- experiments / evaluation
- final inference 

Different resampling and preprocessing pipelines can change internal logits and affects LM acceptance decisions. 

### Configuration Summary

#### LexiCorrect
- `lexicon_term`: list of domain terms (lowercased recommended).
- `num_hypotheses`: *K* hypotheses per segment (N-best). 
- Phonetic replacement:
    - max n-gram window (e.g. 7).
    - dynamic similarity thresholding (length-aware).
    - preserves punctuation and case 
    - optional accent restoration using lexicon map. 

- LLM Re-scoring (optional)   
    - `use_lm`: enable/disable LLM scoring.
    - `deep_lm`: prefix-span search to avoid partial leftover tokens.
    - `lm_threshold`: minimum loss improvement to accept a replacement.
    - `quantization`: `none`, `4bit`, `8bit`.

## Acknowledgments

## Licence 

## Citation
