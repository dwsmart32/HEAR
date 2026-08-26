from .base import BaseHandler


class VoxtralHandler(BaseHandler):
    def get_engine_args(self):
        return {
            "model": self.model_path,
            "tokenizer_mode": self.engine_cfg.get("tokenizer_mode", "mistral"),
            "config_format": self.engine_cfg.get("config_format", "mistral"),
            "load_format": self.engine_cfg.get("load_format", "mistral"),
            "max_model_len": self.engine_cfg.get("max_model_len", 8192),
            "max_num_seqs": self.engine_cfg.get("max_num_seqs", 2),
            "limit_mm_per_prompt": self.engine_cfg.get(
                "limit_mm_per_prompt", {"audio": 1}
            ),
            "enforce_eager": self.engine_cfg.get("enforce_eager", True),
            "enable_chunked_prefill": self.engine_cfg.get(
                "enable_chunked_prefill", False
            ),
        }

    def build_inputs(self, audio_path: str, prompt: str):

        # Voxtral takes interleaved audio and text chunks and converts them to token ids.
        audio_file = self.Audio.from_file(audio_path, strict=False)
        audio_chunk = self.AudioChunk(input_audio=self.RawAudio.from_audio(audio_file))
        text_chunk = self.TextChunk(text=prompt)

        messages = [self.UserMessage(content=[audio_chunk, text_chunk])]
        req = self.ChatCompletionRequest(messages=messages, model=self.model_path)

        # Tokenizing
        encoded = self.tokenizer.encode_chat_completion(req)

        # Convert to the format vLLM expects.
        audios_and_sr = [(au.audio_array, au.sampling_rate) for au in encoded.audios]

        return {
            "prompt_token_ids": encoded.tokens,
            "multi_modal_data": {"audio": audios_and_sr},
        }

    def load_assets(self):
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            from mistral_common.audio import Audio
            from mistral_common.protocol.instruct.chunk import (
                AudioChunk,
                RawAudio,
                TextChunk,
            )
            from mistral_common.protocol.instruct.messages import UserMessage
            from mistral_common.protocol.instruct.request import ChatCompletionRequest

            # Save these classes to self so build_inputs uses them
            self.Audio = Audio
            self.AudioChunk = AudioChunk
            self.RawAudio = RawAudio
            self.TextChunk = TextChunk
            self.UserMessage = UserMessage
            self.ChatCompletionRequest = ChatCompletionRequest

            self.tokenizer = MistralTokenizer.from_hf_hub(self.model_path)
        except ImportError:
            raise ImportError(
                "Voxtral handler requires 'mistral_common'. "
                "pip install mistral-common[audio]"
            )
