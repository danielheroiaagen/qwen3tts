import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os

def test_voice_cloning():
    print("Iniciando prueba de clonación de voz...")
    # Usando el modelo de 0.6B para una prueba más rápida y con menos consumo de memoria
    model_id = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    
    print(f"Cargando modelo {model_id}...")
    try:
        # Intentamos cargar en CUDA si está disponible
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Usando dispositivo: {device}")
        
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=torch.float16 if device == "cuda:0" else torch.float32,
        )
        print("Modelo cargado exitosamente.")
        
        ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_1.wav"
        ref_text  = "甚至出现交易几乎停滞的情况。"

        print("Generando audio clonado...")
        text_to_synthesize = "Hola, esta es una prueba de clonación de voz con el modelo Qwen3-TTS. Estoy verificando si el sistema funciona correctamente."
        
        wavs, sr = model.generate_voice_clone(
            text=text_to_synthesize,
            language="Spanish",
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        
        output_path = "prueba_clonacion.wav"
        sf.write(output_path, wavs[0], sr)
        print(f"¡Clonación exitosa! El audio se ha guardado en: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"Ocurrió un error durante la prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_voice_cloning()
