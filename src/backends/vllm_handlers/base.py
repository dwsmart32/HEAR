from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseHandler(ABC):
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.engine_cfg = self.config.get("engine_args", {})

    @abstractmethod
    def get_engine_args(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def build_inputs(self, audio_path: str, prompt: str) -> Dict[str, Any]:
        pass

    def load_assets(self):
        pass

    def get_audio_token(self, index: int = 0) -> str:
        """Return the model-specific audio placeholder token.
        Default: Qwen2.5/MiDaSheng/FunAudioChat style."""
        return "<|audio_bos|><|AUDIO|><|audio_eos|>"
