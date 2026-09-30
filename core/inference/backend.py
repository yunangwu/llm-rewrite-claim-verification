"""
Inference backend abstraction.

Today: EndpointBackend — wraps AsyncOpenAI pointing at a vllm --serve endpoint.
Future: VllmBackend — loads the model in-process via vllm Python API.

To switch backends, change `mode` in config/inference.yaml:
  mode: endpoint   (current)
  mode: vllm       (TODO)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Minimal async interface used by pipeline.py."""

    @abstractmethod
    async def chat_complete(self, messages: list[dict], **kwargs: Any) -> Any:
        """
        Send a chat completion request.
        `messages` must be a list of OpenAI-style role/content dicts, e.g.
          [{"role": "user", "content": [{type: image_url, ...}, {type: text, ...}]}]
        Returns an openai-compatible ChatCompletion object with:
          .choices[0].message.content
          .choices[0].logprobs.content[0].top_logprobs  (when logprobs=True)
        """
        ...


# ---------------------------------------------------------------------------
# Endpoint backend (current)
# ---------------------------------------------------------------------------


class EndpointBackend(Backend):
    """
    Calls a running vllm --serve (or any OpenAI-compat) endpoint.
    All model-load params are handled by whoever started the server.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "unused",
        timeout: float = 300.0,
        max_retries: int = 3,
    ) -> None:
        import httpx
        from openai import AsyncOpenAI

        self.model = model
        self.max_retries = max_retries
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    async def chat_complete(self, messages: list[dict], **kwargs: Any) -> Any:
        import asyncio as _asyncio

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await self._client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    await _asyncio.sleep(2 ** attempt)  # 1s, 2s, 4s
        raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# vllm Python backend (TODO)
# ---------------------------------------------------------------------------


class VllmBackend(Backend):
    """
    Runs the model in-process via vllm's AsyncLLMEngine.
    Model params (tensor_parallel_size, dtype, …) come from config/models.yaml.

    TODO: implement multimodal input formatting for VL models.
    The interface is already wired — fill in __init__ and chat_complete.
    """

    def __init__(self, model: str, model_cfg: dict[str, Any]) -> None:
        # path: use local path if provided, else HF model ID
        load_path = model_cfg.pop("path", model)
        # from vllm import AsyncLLMEngine, AsyncEngineArgs
        # args = AsyncEngineArgs(model=load_path, **model_cfg)
        # self.engine = AsyncLLMEngine.from_engine_args(args)
        raise NotImplementedError(
            "VllmBackend is not yet implemented. "
            "Set mode: endpoint in config/inference.yaml to use the server endpoint."
        )

    async def chat_complete(self, messages: list[dict], **kwargs: Any) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_backend(model: str, inference_cfg: dict, models_cfg: dict) -> Backend:
    """
    Create the correct backend from config.

    inference_cfg: loaded from config/inference.yaml
    models_cfg:    loaded from config/models.yaml
    """
    mode = inference_cfg.get("mode", "endpoint")
    if mode == "endpoint":
        return EndpointBackend(
            model=model,
            base_url=inference_cfg.get("base_url", "http://localhost:8000/v1"),
            timeout=inference_cfg.get("request_timeout", 300.0),
            max_retries=inference_cfg.get("request_retries", 3),
        )
    if mode == "vllm":
        model_cfg = dict(models_cfg.get(model, {}))  # copy so we can pop 'path'
        return VllmBackend(model=model, model_cfg=model_cfg)
    raise ValueError(f"Unknown backend mode {mode!r}. Choose 'endpoint' or 'vllm'.")
