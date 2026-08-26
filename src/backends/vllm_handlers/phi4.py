import os
import librosa
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest
from .base import BaseHandler


class Phi4Handler(BaseHandler):
    def get_audio_token(self, index: int = 0):
        return f"<|audio_{index + 1}|>"

    def get_engine_args(self):
        model_ref = getattr(self, "local_model_path", self.model_path)
        return {
            "model": model_ref,
            "max_model_len": self.engine_cfg.get("max_model_len", 12800),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 2),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "enable_lora": self.engine_cfg.get("enable_lora", True),
            "max_lora_rank": self.engine_cfg.get("max_lora_rank", 320),
            "trust_remote_code": self.engine_cfg.get("trust_remote_code", True),
            "enforce_eager": self.engine_cfg.get("enforce_eager", False),
        }

    def _get_lora_request(self):
        speech_lora_path = os.path.join(self.local_model_path, "speech-lora")
        if os.path.isdir(speech_lora_path):
            return LoRARequest("speech", 1, speech_lora_path)
        raise FileNotFoundError(f"Speech LoRA not found at {speech_lora_path}")

    def build_inputs(self, audio_path: str, prompt: str):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        full_prompt = f"<|user|><|audio_1|>{prompt}<|end|><|assistant|>"
        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
            "lora_request": self._get_lora_request(),
        }

    def load_assets(self):
        if not os.path.exists(self.model_path):
            print(f"[Phi4Handler] Downloading snapshot for {self.model_path}...")
            self.local_model_path = snapshot_download(self.model_path)
        else:
            self.local_model_path = self.model_path
