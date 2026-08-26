import librosa
from .base import BaseHandler


class AudioFlamingoHandler(BaseHandler):
    def get_audio_token(self, index: int = 0):
        return "<sound>"

    def get_engine_args(self):
        return {
            "model": self.model_path,
            "max_model_len": self.engine_cfg.get("max_model_len", 4096),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 2),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "trust_remote_code": self.engine_cfg.get("trust_remote_code", True),
            "enforce_eager": self.engine_cfg.get("enforce_eager", True),
        }

    def build_inputs(self, audio_path: str, prompt: str):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        audio_placeholder = "<sound>"
        full_prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{audio_placeholder}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
        }
