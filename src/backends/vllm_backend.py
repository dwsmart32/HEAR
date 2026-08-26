# src/backends/vllm_backend.py
import json as json_mod
from typing import Optional, Dict, Any, List, Union
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from .vllm_handlers import get_handler
from .base import BaseBackend

# JSON schema for structured (constrained-decoding) output
_ASR_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "speaker": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["speaker", "text"],
    },
}


class VLLMBackend(BaseBackend):
    def load_model(self, model_name, max_audio_per_prompt: int = 1):
        """
        Initializes the vLLM engine.
        max_audio_per_prompt: maximum number of audio inputs in any single prompt
                             (computed from data before engine init).
        """
        self.model_path = self.config['path']

        print(f"[VLLMBackend] Initializing Handler for: {self.model_path}")
        self.handler = get_handler(model_name, self.model_path, self.config)

        if hasattr(self.handler, 'load_assets'):
            self.handler.load_assets()

        engine_args = self.handler.get_engine_args()

        # Set limit_mm_per_prompt to actual max needed (not hardcoded)
        engine_args["limit_mm_per_prompt"] = {"audio": max(max_audio_per_prompt, 1)}
        print(f"[VLLMBackend] limit_mm_per_prompt set to audio={max_audio_per_prompt}")

        import os as _os
        _tp_env = _os.environ.get("VLLM_TP_OVERRIDE")
        _tp = int(_tp_env) if _tp_env else self.config.get("tensor_parallel_size", 1)
        engine_args.update({
            "tensor_parallel_size": _tp,
            "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0.90),
            "seed": self.config.get("seed", 42),
        })

        print(f"[VLLMBackend] Final Engine Args: {engine_args}")
        self.llm = LLM(**engine_args)

        stop_token_ids = None
        if hasattr(self.handler, 'get_stop_token_ids'):
            stop_token_ids = self.handler.get_stop_token_ids()

        sampling_kwargs = {
            "temperature": self.config.get("temperature", 0.0),
            "max_tokens": self.config.get("max_tokens", 512),
            "stop_token_ids": stop_token_ids,
        }
        if "top_p" in self.config:
            sampling_kwargs["top_p"] = self.config["top_p"]
        if "top_k" in self.config:
            sampling_kwargs["top_k"] = self.config["top_k"]
        if "repetition_penalty" in self.config:
            sampling_kwargs["repetition_penalty"] = self.config["repetition_penalty"]

        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Separate params for structured output: higher max_tokens + JSON schema
        asr_kwargs = {
            **sampling_kwargs,
            "max_tokens": max(self.config.get("max_tokens", 512), 1024),
            "structured_outputs": StructuredOutputsParams(
                json=_ASR_JSON_SCHEMA,
            ),
        }
        self.sampling_params_asr = SamplingParams(**asr_kwargs)

    def generate(self, audio_path: str, prompt: str) -> str:
        """
        Single generation.
        """
        try:
            # Delegate input construction to the handler
            inputs = self.handler.build_inputs(audio_path, prompt)
            outputs = self.llm.generate(
                [inputs],
                sampling_params=self.sampling_params,
                lora_request=inputs.get('lora_request', None),
                use_tqdm=False
            )
            return outputs[0].outputs[0].text.strip()

        except Exception as e:
            print(f"[Error] Generation failed for {audio_path}: {e}")
            raise ValueError(f"Generation failed: {e}")

    def batch_generate(self, items: List[Dict]) -> List[Dict]:
        """Batch generation. Each item carries a single audio_path + instruction."""
        valid_inputs = []
        valid_indices = []
        valid_params = []

        for idx, item in enumerate(items):
            try:
                inp = self.handler.build_inputs(item['audio_path'], item['instruction'])
                valid_inputs.append(inp)
                valid_indices.append(idx)
                valid_params.append(self.sampling_params)
            except Exception as e:
                items[idx]['error'] = f"Build failed: {e}"

        if not valid_inputs:
            return items

        # 2. Handle LoRA (Phi-4 only)
        batch_lora = valid_inputs[0].get('lora_request')

        # 3. Run Inference — pass per-item SamplingParams list
        outputs = self.llm.generate(
            valid_inputs,
            sampling_params=valid_params,
            lora_request=batch_lora,
            use_tqdm=True
        )

        # 4. Map Results
        for i, output in enumerate(outputs):
            original_idx = valid_indices[i]
            items[original_idx]['response'] = output.outputs[0].text.strip()

        return items
