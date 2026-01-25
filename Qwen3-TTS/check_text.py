import torch
from transformers import pipeline

input_wav = "input_reference.wav"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda:0" else torch.float32

print("Transfiriendo...")
asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device, dtype=torch_dtype)
result = asr_pipe(input_wav)
original_text = result["text"].strip()
print(f"Original: {original_text}")

translator = pipeline("translation_es_to_en", model="Helsinki-NLP/opus-mt-es-en", device=device)
trans_result = translator(original_text)
translated_text = trans_result[0]["translation_text"]
print(f"Translated: {translated_text}")
