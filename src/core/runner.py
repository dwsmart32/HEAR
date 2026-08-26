import asyncio
import yaml
from typing import List, Dict
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

class ModelRunner:
    def __init__(
        self,
        model_name: str,
        registry_path="registry.yaml",
        concurrency: int = 10,
        batch_size: int = 1,
    ):
        with open(registry_path) as f:
            registry = yaml.safe_load(f)

        if model_name not in registry.keys():
            raise ValueError(f"Model {model_name} not found.")

        self.config = registry[model_name]
        if self.config.get("disabled", False):
            reason = self.config.get("disabled_reason", "disabled in registry")
            raise RuntimeError(f"Model {model_name} is disabled: {reason}")
        self.backend_type = self.config.get('type', 'local')
        self.concurrency = max(1, concurrency)
        self.batch_size = max(1, batch_size)

        self.backend = self._get_backend(self.config)
        self.model_name = model_name

        # Defer model loading until inference() so we know max_audio_per_prompt
        self._model_loaded = False

    def _get_backend(self, config):
        m_type = config['type']
        if m_type == "huggingface":
            from src.backends.hf_backend import HFBackend
            return HFBackend(config)
        elif m_type == "api":
            from src.backends.api_backend import APIBackend
            return APIBackend(config)
        elif m_type == "vllm":
            from src.backends.vllm_backend import VLLMBackend
            return VLLMBackend(config)
        elif m_type == "external":
            from src.backends.external_backend import ExternalBackend
            return ExternalBackend(config)
        else:
            raise ValueError(f"Unknown type: {m_type}")

    def inference(self, input_data: List[Dict]) -> List[Dict]:
        # Lazy model loading: scan data first to determine max audio count
        if not self._model_loaded and hasattr(self.backend, 'load_model'):
            max_audio = max((item.get("num_audio", 1) for item in input_data), default=1)
            print(f"Loading model: {self.config.get('path', self.model_name)}... "
                  f"(max_audio_per_prompt={max_audio})")
            import inspect
            if 'max_audio_per_prompt' in inspect.signature(self.backend.load_model).parameters:
                self.backend.load_model(self.model_name, max_audio_per_prompt=max_audio)
            else:
                self.backend.load_model(self.model_name)
            self._model_loaded = True

        if self.backend_type == "api":
            print("Detected API Backend: Switching to ASYNC execution mode.")
            return asyncio.run(self._run_async(input_data))

        if hasattr(self.backend, "batch_generate"):
            print(
                f"Detected Non-API Backend with batch_generate: "
                f"running batched inference (batch_size={self.batch_size})."
            )
            return self._run_sync_batched(input_data)

        print("Detected Non-API Backend: Running sequentially.")
        return self._run_sync(input_data)

    def _run_sync(self, input_data: List[Dict]):
        results = []
        for item in tqdm(input_data, desc="Inference"):
            try:
                response = self.backend.generate(item['audio_path'], item['instruction'])
                item['response'] = response
            except Exception as e:
                item['error'] = str(e)
            results.append(item)
        return results

    def _run_sync_batched(self, input_data: List[Dict]):
        results = []
        iterator = range(0, len(input_data), self.batch_size)

        for start_idx in tqdm(iterator, desc=f"Batched Inference (bs={self.batch_size})"):
            batch = input_data[start_idx:start_idx + self.batch_size]

            try:
                results.extend(self.backend.batch_generate(batch))
                continue
            except Exception as e:
                print(
                    f"[Warning] Batch inference failed at index {start_idx}: {e}. "
                    "Falling back to per-sample inference for this batch."
                )

            for item in batch:
                try:
                    # Fallback: still use batch_generate with single-item batch
                    # to preserve multi-audio support
                    processed = self.backend.batch_generate([item])
                    results.extend(processed)
                except Exception as per_item_error:
                    item['error'] = str(per_item_error)
                    results.append(item)

        return results

    async def _run_async(self, input_data: List[Dict]):
        sem = asyncio.Semaphore(self.concurrency)
        async def process_item(item):
            async with sem:
                try:
                    response = await self.backend.generate(item['audio_path'], item['instruction'])
                    item['response'] = response
                except Exception as e:
                    item['error'] = str(e)
                return item

        tasks = [process_item(item) for item in input_data]
        return await tqdm_asyncio.gather(*tasks, desc="Async Inference")
