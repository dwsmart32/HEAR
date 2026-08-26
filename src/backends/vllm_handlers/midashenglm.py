import librosa
from .base import BaseHandler


def _patch_rope_scaling(config):
    """Remove mrope_section from text_config.rope_scaling.

    MiDaShengLM inherits mrope_section from Qwen2.5-Omni base, but
    vLLM's MiDashengLMModel does not implement SupportsMRoPE, causing
    an assertion error at runtime. Stripping the key lets vLLM fall
    back to standard RoPE positions.
    """
    tc = getattr(config, "text_config", None)
    if tc is not None and hasattr(tc, "rope_scaling"):
        rs = tc.rope_scaling
        if isinstance(rs, dict) and "mrope_section" in rs:
            rs.pop("mrope_section")
    return config


class MiDaShengLMHandler(BaseHandler):
    def get_engine_args(self):
        return {
            "model": self.model_path,
            "trust_remote_code": True,
            "max_model_len": self.engine_cfg.get("max_model_len", 4096),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 5),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "hf_overrides": _patch_rope_scaling,
        }

    def build_inputs(self, audio_path: str, prompt: str):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        system_prompt = "You are a helpful language and speech assistant."
        audio_placeholder = "<|audio_bos|><|AUDIO|><|audio_eos|>"

        full_prompt = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{audio_placeholder}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
        }
