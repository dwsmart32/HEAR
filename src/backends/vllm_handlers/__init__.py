from .minicpmo import MiniCPMOHandler, MiniCPMO45Handler
from .phi4 import Phi4Handler
from .voxtral import VoxtralHandler
from .audioflamingo3 import AudioFlamingoHandler
from .qwen import QwenHandler
from .kimi_audio import KimiAudioHandler
from .midashenglm import MiDaShengLMHandler
from .funaudiochat import FunAudioChatHandler


FAMILY_HANDLER_REGISTRY = {
    "minicpm": MiniCPMOHandler,
    "minicpm_o45": MiniCPMO45Handler,
    "phi": Phi4Handler,
    "voxtral": VoxtralHandler,
    "audio_flamingo": AudioFlamingoHandler,
    "qwen": QwenHandler,
    "kimi_audio": KimiAudioHandler,
    "midashenglm": MiDaShengLMHandler,
    "funaudiochat": FunAudioChatHandler,
}

MODEL_FAMILY_MAP = {
    "minicpm-o-2_6-8b": "minicpm",
    "minicpm-o-4_5-9b": "minicpm_o45",

    "phi-4-mm-6b": "phi",

    "voxtral-mini-3b": "voxtral",
    "voxtral-small-24b": "voxtral",

    "audio-flamingo3": "audio_flamingo",

    "qwen2-audio-7b": "qwen",
    "qwen2_5_omni-3b": "qwen",
    "qwen2_5_omni-7b": "qwen",
    "qwen2_5_omni-7b-temp07": "qwen",
    "qwen2_5_omni-7b-temp10": "qwen",
    "qwen3-omni-instruct-30b": "qwen",
    "qwen3-omni-thinking-30b": "qwen",
    "a2r-30b-a3b": "qwen",

    "kimi-audio-7b-instruct": "kimi_audio",

    "midashenglm-7b": "midashenglm",

    "fun-audio-chat-8b": "funaudiochat",
}


def get_handler(model_name: str, model_path: str, config: dict):
    if model_name not in MODEL_FAMILY_MAP:
        raise ValueError(f"Unsupported model: {model_name}")

    family = MODEL_FAMILY_MAP[model_name]

    if family not in FAMILY_HANDLER_REGISTRY:
        raise ValueError(f"No handler for model family: {family}")

    handler_cls = FAMILY_HANDLER_REGISTRY[family]
    return handler_cls(model_path, config)
