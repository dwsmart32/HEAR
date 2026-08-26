import functools
import librosa
from .base import BaseHandler


# Official Qwen audio/omni baseline HF model IDs.
# Exact-match; anything else (a fine-tune, a local merged checkpoint) is resolved
# by reading its config.json instead of by guessing from the path string.
QWEN_BASELINE_HF_IDS = frozenset({
    "Qwen/Qwen2.5-Omni-3B",
    "Qwen/Qwen2.5-Omni-7B",
    "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "Qwen/Qwen3-Omni-30B-A3B-Thinking",
    "Qwen/Qwen2-Audio-7B-Instruct",
})


@functools.lru_cache(maxsize=32)
def qwen_family(model_path: str) -> str:
    """Return "qwen3_omni" | "qwen2_5_omni" | "qwen2_audio" | "unknown".

    Read from the model's own config.json (`model_type` / `architectures`), not from
    the path. A fine-tune published under any name resolves the same as its base, so
    it gets the right audio placeholder token and context length.
    """
    model_type, arch = "", ""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = (getattr(cfg, "model_type", "") or "").lower()
        arch = " ".join(getattr(cfg, "architectures", None) or []).lower()
    except Exception:
        try:                       # config.json alone, no transformers class needed
            import json, os
            from huggingface_hub import hf_hub_download
            path = (os.path.join(model_path, "config.json") if os.path.isdir(model_path)
                    else hf_hub_download(model_path, "config.json"))
            with open(path) as f:
                cfg = json.load(f)
            model_type = str(cfg.get("model_type", "")).lower()
            arch = " ".join(cfg.get("architectures", []) or []).lower()
        except Exception:
            pass

    blob = model_type + " " + arch
    if "qwen3_omni" in blob or "qwen3omni" in blob:
        return "qwen3_omni"
    if "qwen2_5_omni" in blob or "qwen2_5omni" in blob:
        return "qwen2_5_omni"
    if "qwen2_audio" in blob or "qwen2audio" in blob:
        return "qwen2_audio"

    low = model_path.lower()       # last resort, so an offline path still works
    if any(k in low for k in ("qwen3-omni", "qwen3_omni", "qwen3_30b")):
        return "qwen3_omni"
    if "qwen2-audio" in low:
        return "qwen2_audio"
    if any(k in low for k in ("qwen2.5", "qwen25", "qwen2_5")):
        return "qwen2_5_omni"
    return "unknown"


class QwenHandler(BaseHandler):
    def get_audio_token(self, index: int = 0):
        fam = qwen_family(self.model_path)
        if fam == "qwen3_omni":
            return "<|audio_start|><|audio_pad|><|audio_end|>"
        if fam == "qwen2_audio":
            return f"Audio {index + 1}: <|audio_bos|><|AUDIO|><|audio_eos|>"
        return "<|audio_bos|><|AUDIO|><|audio_eos|>"

    # Multi-audio: uses base handler's inline replacement.
    # vLLM matches audio tokens to audio_list by order of appearance in prompt.

    def get_engine_args(self):
        # Default config per model type
        fam = qwen_family(self.model_path)
        max_len = 8192  # Default for Qwen2-Audio
        if fam == "qwen3_omni":
            max_len = 12800  # Qwen3-Omni defaults to higher context

        return {
            "model": self.model_path,
            "max_model_len": self.engine_cfg.get("max_model_len", max_len),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 5),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "trust_remote_code": self.engine_cfg.get("trust_remote_code", True),
        }

    def build_inputs(self, audio_path: str, prompt: str):
        # All Qwen audio models take 16 kHz mono.
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Family comes from config.json, not from the path string, so a
        # fine-tune published under any name is prompted like its base model.
        fam = qwen_family(self.model_path)
        if fam == "unknown":
            print(
                f"[Warning] Could not determine the Qwen family of "
                f"{self.model_path} from its config.json; using the "
                f"Qwen2.5-Omni prompt format. Set `qwen_family` in the "
                f"registry entry to override."
            )
            fam = self.config.get("qwen_family", "qwen2_5_omni")

        # Qwen's official default system prompt.
        omni_system_prompt = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
            "Group, capable of perceiving auditory and visual inputs, as well as "
            "generating text and speech."
        )

        # System prompt policy, explicit rather than inferred:
        #   registry `system_prompt: "..."`  -> use that string
        #   registry `system_prompt: null`   -> omit the system turn entirely
        #   absent                           -> Qwen's official default, with
        #                                       two exceptions: an official
        #                                       Qwen3-Omni base checkpoint omits
        #                                       it (the model card says to for
        #                                       benchmark evaluation), and
        #                                       Qwen2-Audio uses the generic
        #                                       assistant prompt it ships with.
        is_baseline = self.model_path in QWEN_BASELINE_HF_IDS
        if "system_prompt" in self.config:
            system_prompt = self.config["system_prompt"]
        elif fam == "qwen3_omni" and is_baseline:
            system_prompt = None
        elif fam == "qwen2_audio":
            system_prompt = "You are a helpful assistant."
        else:
            system_prompt = omni_system_prompt

        if fam == "qwen3_omni":
            audio_tag, sep = "<|audio_start|><|audio_pad|><|audio_end|>", ""
        elif fam == "qwen2_audio":
            audio_tag, sep = "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>", "\n"
        else:
            audio_tag, sep = "<|audio_bos|><|AUDIO|><|audio_eos|>", "\n"

        head = (f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                if system_prompt else "")
        full_prompt = (
            f"{head}"
            f"<|im_start|>user\n{audio_tag}{sep}{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
        }
