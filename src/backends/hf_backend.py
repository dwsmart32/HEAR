# src/backends/hf_backend.py
import io
import os
import wave
import torch
import librosa
import numpy as np
from typing import Any, Dict, List, Optional
from .base import BaseBackend


def save_temp_wav(audio_array, sr=16000):
    """Save audio array to a temp wav file, return path."""
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        pcm = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    tmp.write(buf.getvalue())
    tmp.flush()
    return tmp.name


_GEMMA4_MAX_AUDIO_SEC = 30


def _chunk_and_save_audio(audio_path, sr=16000, max_seconds=_GEMMA4_MAX_AUDIO_SEC):
    """Load audio, split into ≤max_seconds chunks, save as temp wavs.
    Returns list of temp file paths (caller must clean up)."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    max_samples = int(max_seconds * sr)
    if len(y) <= max_samples:
        return [], audio_path  # no chunking needed, use original path
    paths = []
    for start in range(0, len(y), max_samples):
        chunk = y[start:start + max_samples]
        paths.append(save_temp_wav(chunk, sr))
    return paths, None  # chunked paths, no original


class HFBackend(BaseBackend):
    def load_model(self, model_name: Optional[str] = None):
        model_path = self.config.get("path", model_name)
        hf_model_type = self.config.get("hf_model_type", "generic")

        if hf_model_type == "audioflamingo3":
            self._load_audioflamingo3(model_path)
        elif hf_model_type == "gemma4":
            self._load_gemma4(model_path)
        elif hf_model_type == "kimi_audio":
            self._load_kimi_audio(model_path)
        elif hf_model_type == "minicpmo":
            self._load_minicpmo(model_path)
        else:
            raise ValueError(f"Unknown hf_model_type: {hf_model_type}")

    def _load_gemma4(self, model_path: str):
        from transformers import AutoProcessor, AutoModelForMultimodalLM

        self.hf_model_type = "gemma4"
        self.gemma4_processor = AutoProcessor.from_pretrained(model_path)
        self.gemma4_model = AutoModelForMultimodalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def _load_audioflamingo3(self, model_path: str):
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        self.hf_model_type = "audioflamingo3"
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )

    def _load_minicpmo(self, model_path: str):
        from transformers import AutoModel, AutoTokenizer

        self.hf_model_type = "minicpmo"
        self.minicpmo_model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=False,
        )
        self.minicpmo_model = self.minicpmo_model.eval().cuda()
        self.minicpmo_tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

    def _load_kimi_audio(self, model_path: str):
        from kimia_infer.api.kimia import KimiAudio

        self.hf_model_type = "kimi_audio"
        self.kimi_model = KimiAudio(model_path=model_path, load_detokenizer=False)

    def generate(self, audio_path: str, prompt: str) -> str:
        if self.hf_model_type == "audioflamingo3":
            return self._generate_audioflamingo3(audio_path, prompt)
        elif self.hf_model_type == "gemma4":
            return self._generate_gemma4(audio_path, prompt)
        elif self.hf_model_type == "kimi_audio":
            return self._generate_kimi_audio(audio_path, prompt)
        elif self.hf_model_type == "minicpmo":
            return self._generate_minicpmo(audio_path, prompt)
        else:
            raise ValueError(f"Unknown hf_model_type: {self.hf_model_type}")

    def batch_generate(self, items: List[Dict]) -> List[Dict]:
        for item in items:
            try:
                item["response"] = self.generate(item["audio_path"], item["instruction"])
            except Exception as e:
                item["error"] = str(e)
        return items

    def _build_generate_sampling_kwargs(self, default_max_tokens: int = 512) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.config.get("max_tokens", default_max_tokens),
        }
        use_sampling = self.config.get("do_sample")
        if use_sampling is None:
            use_sampling = (
                self.config.get("temperature", 0.0) > 0
                or "top_p" in self.config
                or "top_k" in self.config
            )

        if use_sampling:
            kwargs["do_sample"] = True
            if "temperature" in self.config:
                kwargs["temperature"] = self.config["temperature"]
            if "top_p" in self.config:
                kwargs["top_p"] = self.config["top_p"]
            if "top_k" in self.config:
                kwargs["top_k"] = self.config["top_k"]
        return kwargs

    def _generate_gemma4(self, audio_path: str, prompt: str) -> str:
        chunk_paths, orig = _chunk_and_save_audio(audio_path)
        content: List[Dict[str, Any]] = []
        if orig:
            content.append({"type": "audio", "audio": orig})
        else:
            for cp in chunk_paths:
                content.append({"type": "audio", "audio": cp})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]
        try:
            inputs = self.gemma4_processor.apply_chat_template(
                messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
            ).to(self.gemma4_model.device)
            input_len = inputs["input_ids"].shape[-1]
            outputs = self.gemma4_model.generate(
                **inputs, **self._build_generate_sampling_kwargs())
            response = self.gemma4_processor.decode(
                outputs[0][input_len:], skip_special_tokens=True)
            return response.strip()
        finally:
            for p in chunk_paths:
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    def _generate_audioflamingo3(self, audio_path: str, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "path": audio_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(device=self.model.device, dtype=torch.bfloat16)

        outputs = self.model.generate(
            **inputs,
            **self._build_generate_sampling_kwargs(default_max_tokens=256),
        )
        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        return decoded[0].strip()

    def _generate_minicpmo(self, audio_path: str, prompt: str) -> str:
        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        thinking = bool(self.config.get("enable_thinking", False))
        msgs = []
        if thinking:
            msgs.append({
                "role": "system",
                "content": "You must always think step-by-step inside <think>...</think> tags first, then give your final answer.",
            })
        msgs.append({"role": "user", "content": [audio_input, prompt]})
        temperature = self.config.get("temperature", 0.0)
        use_sampling = temperature > 0
        chat_kwargs = {
            "msgs": msgs,
            "tokenizer": self.minicpmo_tokenizer,
            "sampling": use_sampling,
            "max_new_tokens": self.config.get("max_tokens", 256),
            "use_tts_template": False,
            "generate_audio": False,
        }
        if "enable_thinking" in self.config:
            chat_kwargs["enable_thinking"] = thinking
        if use_sampling:
            chat_kwargs["temperature"] = temperature
            if "top_p" in self.config:
                chat_kwargs["top_p"] = self.config["top_p"]
            if "top_k" in self.config:
                chat_kwargs["top_k"] = self.config["top_k"]
        res = self.minicpmo_model.chat(**chat_kwargs)
        return res.strip()

    def _build_kimi_sampling_params(self) -> Dict[str, Any]:
        return {
            "audio_temperature": self.config.get("audio_temperature", 0.0),
            "audio_top_k": self.config.get("audio_top_k", 5),
            "text_temperature": self.config.get("text_temperature", 0.0),
            "text_top_k": self.config.get("text_top_k", 5),
            "audio_repetition_penalty": self.config.get("audio_repetition_penalty", 1.0),
            "audio_repetition_window_size": self.config.get("audio_repetition_window_size", 64),
            "text_repetition_penalty": self.config.get("text_repetition_penalty", 1.0),
            "text_repetition_window_size": self.config.get("text_repetition_window_size", 16),
            "max_new_tokens": self.config.get("max_tokens", -1),
        }

    def _generate_kimi_audio(self, audio_path: str, prompt: str) -> str:
        messages = [
            {"role": "user", "message_type": "audio", "content": audio_path},
            {"role": "user", "message_type": "text", "content": prompt},
        ]

        _, text_output = self.kimi_model.generate(
            messages, **self._build_kimi_sampling_params(), output_type="text"
        )
        return text_output.strip()
