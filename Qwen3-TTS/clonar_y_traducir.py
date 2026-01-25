import torch
import soundfile as sf
import os
import gc
from transformers import pipeline
from qwen_tts import Qwen3TTSModel

def clone_and_translate():
    input_wav = "input_reference.wav"
    output_wav = "voz_clonada_ingles.wav"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda:0" else torch.float32

    print(f"--- Paso 1: Transcribiendo el audio original ({input_wav}) ---")
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=device,
        dtype=torch_dtype
    )
    
    result = asr_pipe(input_wav)
    original_text = result["text"].strip()
    print(f"Texto original detectado: {original_text}")
    
    del asr_pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    print(f"\n--- Paso 2: Traduciendo al inglés americano ---")
    translator = pipeline(
        "translation_es_to_en",
        model="Helsinki-NLP/opus-mt-es-en",
        device=device
    )
    
    translation_result = translator(original_text)
    translated_text = translation_result[0]["translation_text"]
    print(f"Texto traducido: {translated_text}")
    
    del translator
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

    print(f"\n--- Paso 3: Clonando voz con Qwen3-TTS ---")
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    print(f"Cargando modelo Qwen3-TTS {model_id}...")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch_dtype,
    )

    print("Generando audio con tu voz clonada en inglés...")
    wavs, sr = model.generate_voice_clone(
        text=translated_text,
        language="English",
        ref_audio=input_wav,
        ref_text=original_text,
    )
    
    sf.write(output_wav, wavs[0], sr)
    print(f"\n¡Proceso completado con éxito!")
    print(f"El audio traducido con tu voz se ha guardado en: {os.path.abspath(output_wav)}")

if __name__ == "__main__":
    try:
        clone_and_translate()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
