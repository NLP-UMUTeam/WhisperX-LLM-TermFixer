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