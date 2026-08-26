"""API backend (Google Gemini / OpenAI GPT).

Prompt layout: [MAIN_AUDIO, INSTRUCTION_TEXT]

All audio types (A/B/C) are pre-built into a single WAV file,
so the backend always sends exactly one audio + one text instruction.
"""

import os
import base64
import json
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.core.schema import BenchmarkResponse
from src.backends.base import BaseBackend

load_dotenv()


class APIBackend(BaseBackend):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.provider = config['provider']
        self.model_name = config['model_name']

        if self.provider == "google":
            from google import genai
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        elif self.provider == "openai":
            from openai import AsyncOpenAI
            env_key = config.get("env_key", "OPENAI_API_KEY")
            api_key = os.getenv(env_key)
            if not api_key:
                raise RuntimeError(f"Env var {env_key} is not set.")
            print(f"[APIBackend] openai using env key: {env_key}")
            self.client = AsyncOpenAI(api_key=api_key)

    def load_model(self, model_name: Optional[str] = None) -> None:
        return None

    def batch_generate(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        async def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
            try:
                item["response"] = await self.generate(
                    audio_path=item["audio_path"],
                    instruction=item["instruction"],
                )
            except Exception as e:
                item["error"] = str(e)
            return item

        async def run_batch() -> List[Dict[str, Any]]:
            tasks = [process_item(item) for item in inputs]
            return await asyncio.gather(*tasks)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run_batch())

        raise RuntimeError(
            "APIBackend.batch_generate cannot run inside an active event loop."
        )

    async def generate(self, audio_path: str, instruction: str) -> Dict:
        if self.provider == "google":
            return await self._call_gemini(audio_path, instruction)
        if self.provider == "openai":
            return await self._call_gpt(audio_path, instruction)
        return {"error": "Unknown provider"}

    # ──────────────────────────────────────────────────────────────────
    # Gemini
    # ──────────────────────────────────────────────────────────────────
    async def _call_gemini(self, audio_path, instruction):
        from google.genai import types

        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        contents = [
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            instruction,
        ]

        timeout_s = float(self.config.get("request_timeout_s", 45))
        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=self.model_name,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": BenchmarkResponse,
                },
            ),
            timeout=timeout_s,
        )
        return json.loads(response.text)

    # ──────────────────────────────────────────────────────────────────
    # OpenAI (gpt-4o-audio / gpt-4o)
    # ──────────────────────────────────────────────────────────────────
    async def _call_gpt(self, audio_path, instruction):
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")

        content = [
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": instruction},
        ]

        messages = [{"role": "user", "content": content}]

        if "audio" in self.model_name:
            completion = await self.client.chat.completions.create(
                model=self.model_name, messages=messages
            )
            return completion.choices[0].message.content
        else:
            completion = await self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=BenchmarkResponse,
            )
            return completion.choices[0].message.parsed.model_dump()
