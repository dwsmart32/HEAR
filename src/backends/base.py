from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

class BaseBackend(ABC):
    def __init__(self, config: Dict):
        self.config = config

    def load_model(self, model_name: Optional[str] = None) -> None:
        # Some backends (API) do not require explicit model loading.
        return None

    @abstractmethod
    def generate(self, audio_path: str, prompt: str) -> str:
        pass

    def batch_generate(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for item in inputs:
            try:
                item["response"] = self.generate(item["audio_path"], item["instruction"])
            except Exception as e:
                item["error"] = str(e)
            results.append(item)
        return results
