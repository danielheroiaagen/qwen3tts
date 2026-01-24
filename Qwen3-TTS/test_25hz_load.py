
import torch
from qwen_tts.core.tokenizer_25hz.configuration_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Config
from qwen_tts.core.tokenizer_25hz.modeling_qwen3_tts_tokenizer_v1 import Qwen3TTSTokenizerV1Model

def test_load():
    try:
        config = Qwen3TTSTokenizerV1Config()
        model = Qwen3TTSTokenizerV1Model(config)
        print("Successfully instantiated 25Hz model with default config")
    except Exception as e:
        print(f"Error instantiating model: {e}")

if __name__ == "__main__":
    test_load()
