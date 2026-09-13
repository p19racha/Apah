"""Thin HTTP client wrapper around the Apah server API."""

import json
from typing import Any, Dict, Generator, List, Optional
import httpx


class ApahClientError(Exception):
    """Base exception class for Apah client errors."""
    pass


class ServerNotRunningError(ApahClientError):
    """Raised when the Apah server is unreachable or connection is refused."""
    pass


class ModelNotLoadedError(ApahClientError):
    """Raised when an operation requires a loaded model but none is active."""
    pass


class ApahAPIError(ApahClientError):
    """Raised when the Apah server returns an HTTP error status code."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ApahClient:
    """HTTP client wrapper communicating with an active Apah server."""

    def __init__(self, base_url: str = "http://localhost:11500", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        """Check server liveness (/health)."""
        url = f"{self.base_url}/health"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.ConnectRefusedError) as e:
            raise ServerNotRunningError(
                f"Could not connect to Apah server at {self.base_url}. Run 'apah serve' first."
            ) from e
        except httpx.HTTPStatusError as e:
            raise ApahAPIError(f"Health check failed: {e.response.text}", status_code=e.response.status_code) from e
        except Exception as e:
            raise ApahAPIError(f"Unexpected error calling health check: {e}") from e

    def ps(self) -> List[Dict[str, Any]]:
        """Get status of currently loaded model and scheduler (/ps)."""
        url = f"{self.base_url}/ps"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.ConnectRefusedError) as e:
            raise ServerNotRunningError(
                f"Could not connect to Apah server at {self.base_url}. Run 'apah serve' first."
            ) from e
        except httpx.HTTPStatusError as e:
            raise ApahAPIError(f"Failed to fetch process status: {e.response.text}", status_code=e.response.status_code) from e
        except Exception as e:
            raise ApahAPIError(f"Unexpected error fetching process status: {e}") from e

    def gpu(self) -> List[Dict[str, Any]]:
        """Get real-time NVML hardware stats for visible GPUs (/gpu)."""
        url = f"{self.base_url}/gpu"
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.ConnectRefusedError) as e:
            raise ServerNotRunningError(
                f"Could not connect to Apah server at {self.base_url}. Run 'apah serve' first."
            ) from e
        except httpx.HTTPStatusError as e:
            raise ApahAPIError(f"Failed to fetch GPU hardware stats: {e.response.text}", status_code=e.response.status_code) from e
        except Exception as e:
            raise ApahAPIError(f"Unexpected error fetching GPU stats: {e}") from e

    def load(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        gpu_mem_fraction: float = 0.9,
        tp_world_size: int = 1,
    ) -> Dict[str, Any]:
        """Load a model into the server runtime (/load)."""
        url = f"{self.base_url}/load"
        payload = {
            "model_path": model_path,
            "dtype": dtype,
            "gpu_mem_fraction": gpu_mem_fraction,
            "tp_world_size": tp_world_size,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("error", resp.text)
                    except Exception:
                        err_msg = resp.text
                    raise ApahAPIError(f"Load model failed: {err_msg}", status_code=resp.status_code)
                return resp.json()
        except (httpx.ConnectError, httpx.ConnectRefusedError) as e:
            raise ServerNotRunningError(
                f"Could not connect to Apah server at {self.base_url}. Run 'apah serve' first."
            ) from e
        except ApahAPIError:
            raise
        except Exception as e:
            raise ApahAPIError(f"Unexpected error loading model: {e}") from e

    def unload(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Unload currently loaded model from server (/unload)."""
        url = f"{self.base_url}/unload"
        payload = {"model_name": model_name} if model_name else {}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code != 200:
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("error", resp.text)
                    except Exception:
                        err_msg = resp.text
                    raise ApahAPIError(f"Unload model failed: {err_msg}", status_code=resp.status_code)
                return resp.json()
        except (httpx.ConnectError, httpx.ConnectRefusedError) as e:
            raise ServerNotRunningError(
                f"Could not connect to Apah server at {self.base_url}. Run 'apah serve' first."
            ) from e
        except ApahAPIError:
            raise
        except Exception as e:
            raise ApahAPIError(f"Unexpected error unloading model: {e}") from e

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Send a chat completion request to the server (/v1/chat/completions)."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 400:
                    try:
                        err_json = resp.json()
                        err_detail = err_json.get("error", {})
                        if isinstance(err_detail, dict) and err_detail.get("code") == "model_not_loaded":
                            raise ModelNotLoadedError("No model is currently loaded on the server.")
                        msg = err_detail.get("message") if isinstance(err_detail, dict) else str(err_detail)
                    except (ValueError, KeyError):
                        msg = resp.text
                    raise ModelNotLoadedError(f"Bad request: {msg}")
                elif resp.status_code != 200:
                    try:
                        err_json = resp.json()
                        err_msg = err_json.get("error", resp.text)
                    except Exception:
                        err_msg = resp.text
                    raise ApahAPIError(f"Chat completion failed: {err_msg}", status_code=resp.status_code)
                return resp.json()
        except (httpx.ConnectError, httpx.ConnectRefusedError) as e:
            raise ServerNotRunningError(
                f"Could not connect to Apah server at {self.base_url}. Run 'apah serve' first."
            ) from e
        except (ModelNotLoadedError, ApahAPIError):
            raise
        except Exception as e:
            raise ApahAPIError(f"Unexpected error during chat completion: {e}") from e

    def chat_completion_stream(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Stream tokens from a chat completion SSE response."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        try:
            with httpx.Client(timeout=self.timeout) as client:
                with client.stream("POST", url, json=payload) as response:
                    if response.status_code == 400:
                        response.read()
                        try:
                            err_json = response.json()
                            err_detail = err_json.get("error", {})
                            if isinstance(err_detail, dict) and err_detail.get("code") == "model_not_loaded":
                                raise ModelNotLoadedError("No model is currently loaded on the server.")
                            msg = err_detail.get("message") if isinstance(err_detail, dict) else str(err_detail)
                        except Exception:
                            msg = response.text
                        raise ModelNotLoadedError(f"Bad request: {msg}")
                    elif response.status_code != 200:
                        response.read()
                        try:
                            err_json = response.json()
                            err_msg = err_json.get("error", response.text)
                        except Exception:
                            err_msg = response.text
                        raise ApahAPIError(f"Streaming failed: {err_msg}", status_code=response.status_code)

                    for line in response.iter_lines():
                        if not line:
                            continue
                        line = line.strip()
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data_str)
                                choices = chunk.get("choices", [])
                                if choices:
                                    delta = choices[0].get("delta", {})
                                    content = delta.get("content")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                continue
        except (httpx.ConnectError, httpx.ConnectRefusedError) as e:
            raise ServerNotRunningError(
                f"Could not connect to Apah server at {self.base_url}. Run 'apah serve' first."
            ) from e
        except (ModelNotLoadedError, ApahAPIError):
            raise
        except Exception as e:
            raise ApahAPIError(f"Unexpected error during streaming chat completion: {e}") from e
