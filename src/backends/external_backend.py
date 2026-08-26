# src/backends/external_backend.py
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseBackend

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ExternalBackend(BaseBackend):
    """Backend that delegates inference to an external script via JSON-line IPC.

    The external script (bridge) is launched as a subprocess.  Communication
    follows a simple protocol:

    1. On startup the bridge prints ``{"status": "ready"}`` to stdout.
    2. For each sample the backend writes a JSON object
       ``{"audio_path": "...", "prompt": "..."}`` followed by a newline.
    3. The bridge processes the sample and replies with a single JSON line
       ``{"response": "..."}``  (or ``{"error": "..."}`` on failure).
    4. When stdin is closed the bridge exits.
    """

    def __init__(self, config: Dict):
        super().__init__(config)
        self._proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Model loading  (= launching the bridge subprocess)
    # ------------------------------------------------------------------
    def load_model(self, model_name: Optional[str] = None) -> None:
        working_dir = self.config.get("working_dir", ".")
        working_dir = (PROJECT_ROOT / working_dir).resolve()

        env_path = self.config.get("env")
        if env_path is not None:
            python = str(
                (PROJECT_ROOT / "model_env" / env_path / ".venv" / "bin" / "python").resolve()
            )
        else:
            python = sys.executable

        command = self.config.get("command", "python inference_bridge.py")
        cmd_parts = command.split()
        # Replace bare "python" with the resolved interpreter.
        if cmd_parts[0] == "python":
            cmd_parts[0] = python

        # Pass registry-level extra args as env vars so the bridge can read them.
        extra_env: Dict[str, str] = {}
        for key in ("checkpoint", "train_config", "inference_config"):
            val = self.config.get(key)
            if val is not None:
                extra_env[f"HARNESS_{key.upper()}"] = str((PROJECT_ROOT / val).resolve())

        import os

        env = {**os.environ, **extra_env}

        self._proc = subprocess.Popen(
            cmd_parts,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(working_dir),
            env=env,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Wait for the bridge to signal readiness.
        ready_line = self._proc.stdout.readline()
        if not ready_line:
            stderr_snippet = self._proc.stderr.read(2000)
            raise RuntimeError(
                f"External bridge failed to start.\nstderr:\n{stderr_snippet}"
            )

        try:
            msg = json.loads(ready_line)
        except json.JSONDecodeError:
            raise RuntimeError(
                f"External bridge sent non-JSON ready line: {ready_line!r}"
            )

        if msg.get("status") != "ready":
            raise RuntimeError(f"External bridge did not report ready: {msg}")

        print(f"[ExternalBackend] Bridge is ready (pid={self._proc.pid})")

    def _sampling_params(self) -> Dict[str, Any]:
        """Extract sampling params from registry config for the bridge."""
        params: Dict[str, Any] = {}
        for key in ("temperature", "top_p", "top_k", "max_tokens"):
            if key in self.config:
                params[key] = self.config[key]
        return params

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, audio_path: str, prompt: str) -> str:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("External bridge process is not running.")

        request = json.dumps({
            "audio_path": audio_path,
            "prompt": prompt,
            **self._sampling_params(),
        })
        self._proc.stdin.write(request + "\n")
        self._proc.stdin.flush()

        response_line = self._proc.stdout.readline()
        if not response_line:
            stderr_snippet = self._proc.stderr.read(2000)
            raise RuntimeError(
                f"External bridge returned empty response.\nstderr:\n{stderr_snippet}"
            )

        result = json.loads(response_line)
        if "error" in result:
            raise RuntimeError(f"External bridge error: {result['error']}")

        return result["response"]

    # ------------------------------------------------------------------
    # Batch generation (concurrent via bridge batch protocol)
    # ------------------------------------------------------------------
    def batch_generate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("External bridge process is not running.")

        sampling = self._sampling_params()
        batch_request = {
            "batch": [
                {
                    "id": i,
                    "audio_path": item["audio_path"],
                    "prompt": item["instruction"],
                    **sampling,
                }
                for i, item in enumerate(items)
            ]
        }

        self._proc.stdin.write(json.dumps(batch_request) + "\n")
        self._proc.stdin.flush()

        response_line = self._proc.stdout.readline()
        if not response_line:
            stderr_snippet = self._proc.stderr.read(2000)
            raise RuntimeError(
                f"External bridge returned empty batch response.\nstderr:\n{stderr_snippet}"
            )

        result = json.loads(response_line)
        if "batch" not in result:
            raise RuntimeError(f"Expected batch response, got: {result}")

        for r in result["batch"]:
            idx = r["id"]
            if "error" in r:
                items[idx]["error"] = r["error"]
            else:
                items[idx]["response"] = r["response"]

        return items

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self):
        if self._proc is not None and self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
