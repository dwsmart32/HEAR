import librosa
from transformers import AutoTokenizer
from .base import BaseHandler


class MiniCPMOHandler(BaseHandler):
    """Handler for MiniCPM-o 2.6 (Qwen2 backbone)."""

    def get_audio_token(self, index: int = 0):
        return "(<audio>./</audio>)"

    def load_assets(self):
        # MiniCPM requires the chat template to be applied through its tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.engine_cfg.get("trust_remote_code", True),
        )
        self.stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        self.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(i) for i in self.stop_tokens
        ]

    def get_engine_args(self):
        return {
            "model": self.model_path,
            "max_model_len": self.engine_cfg.get("max_model_len", 4096),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 2),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "trust_remote_code": self.engine_cfg.get("trust_remote_code", True),
            "enforce_eager": self.engine_cfg.get("enforce_eager", False),
        }

    def get_stop_token_ids(self):
        return self.stop_token_ids

    def build_inputs(self, audio_path: str, prompt: str):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        system_prompt = (
            "You are a helpful assistant. "
            "You can accept audio and text input and output voice and text."
        )
        audio_placeholder = "(<audio>./</audio>)"
        messages = [
            {"role": "user", "content": f"{system_prompt}\n{audio_placeholder}\n{prompt}"},
        ]

        audio_chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=audio_chat_template,
        )

        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
        }


class MiniCPMO45Handler(BaseHandler):
    """Handler for MiniCPM-o 4.5 (Qwen3-8B backbone, 9B params)."""

    def get_audio_token(self, index: int = 0):
        return "(<audio>./</audio>)"

    def load_assets(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.engine_cfg.get("trust_remote_code", True),
        )
        self.stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        self.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(i) for i in self.stop_tokens
        ]

    def get_engine_args(self):
        return {
            "model": self.model_path,
            "max_model_len": self.engine_cfg.get("max_model_len", 4096),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 2),
            "max_num_batched_tokens": self.engine_cfg.get(
                "max_num_batched_tokens", 2048
            ),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "trust_remote_code": self.engine_cfg.get("trust_remote_code", True),
            "enforce_eager": self.engine_cfg.get("enforce_eager", False),
            "gpu_memory_utilization": self.engine_cfg.get(
                "gpu_memory_utilization", 0.9
            ),
        }

    def get_stop_token_ids(self):
        return self.stop_token_ids

    def build_inputs(self, audio_path: str, prompt: str):
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        audio_placeholder = "(<audio>./</audio>)"
        messages = [
            {"role": "user", "content": f"{audio_placeholder}\n{prompt}"},
        ]

        # MiniCPM-o 4.5: use tokenizer's built-in chat template (no TTS tokens)
        full_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": full_prompt,
            "multi_modal_data": {"audio": (y, sr)},
        }
