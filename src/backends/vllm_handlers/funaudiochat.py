import librosa
from .base import BaseHandler


class FunAudioChatHandler(BaseHandler):
    def get_engine_args(self):
        return {
            "model": self.model_path,
            "max_model_len": self.engine_cfg.get("max_model_len", 4096),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 2),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "enforce_eager": self.engine_cfg.get("enforce_eager", True),
        }

    def build_inputs(self, audio_path: str, prompt: str):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
        full_prompt = f"{audio_placeholder}{prompt}"

        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
        }
