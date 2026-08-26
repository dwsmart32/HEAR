import librosa
from .base import BaseHandler


class KimiAudioHandler(BaseHandler):
    def get_audio_token(self, index: int = 0):
        return "<|im_kimia_audio|>"

    def get_engine_args(self):
        return {
            "model": self.model_path,
            "trust_remote_code": self.engine_cfg.get("trust_remote_code", True),
            "max_model_len": self.engine_cfg.get("max_model_len", 4096),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 2),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
        }

    def get_stop_token_ids(self):
        # Kimi-Audio reference example uses EOS token id 151644.
        return self.config.get("stop_token_ids", [151644])

    def build_inputs(self, audio_path: str, prompt: str):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        question = prompt if prompt else "Please transcribe the audio"
        full_prompt = f"<|im_kimia_text_blank|>{question}"

        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
        }
