"""Apah inference runtime for single-request text generation with manual KV caching."""

from dataclasses import dataclass
import logging
import time
from typing import Generator, List, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from apah.engine.model_loader import load_model

logger = logging.getLogger("apah.engine.runtime")


@dataclass
class GenerationMetrics:
    """Metrics recorded during a generation run."""

    ttft_sec: float = 0.0  # Time to first token (seconds)
    total_time_sec: float = 0.0  # Total generation elapsed time (seconds)
    tokens_per_sec: float = 0.0  # Generation throughput (tokens/sec)
    generated_tokens: int = 0  # Total new tokens generated
    peak_gpu_memory_mb: float = 0.0  # Peak GPU memory allocated during generation (MB)


class ApahRuntime:
    """Runtime wrapping a loaded Causal LM and Tokenizer for manual KV-cache token generation."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
    ) -> None:
        """Initialize ApahRuntime.

        Args:
            model: Loaded Hugging Face PreTrainedModel.
            tokenizer: Loaded Hugging Face PreTrainedTokenizerBase.
            device: Target execution device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.last_metrics: Optional[GenerationMetrics] = None

    @classmethod
    def from_model_path(
        cls,
        model_path: str,
        dtype: str = "bfloat16",
        device: str = "cuda",
        tp_world_size: int = 1,
    ) -> "ApahRuntime":
        """Factory method to load a model and initialize ApahRuntime."""
        model, tokenizer = load_model(
            model_path=model_path, dtype=dtype, device=device, tp_world_size=tp_world_size
        )
        runtime = cls(model=model, tokenizer=tokenizer, device=device)
        runtime.tp_world_size = tp_world_size
        return runtime

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """Select next token id using greedy decoding or temperature sampling.

        Args:
            logits: Logits tensor of shape (1, vocab_size).
            temperature: Sampling temperature.
            do_sample: If True, perform temperature sampling. If False, greedy decoding (argmax).

        Returns:
            Tensor of shape (1, 1) containing the next token id.
        """
        if not do_sample or temperature <= 0.0:
            # Greedy decoding
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Temperature sampling
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: Optional[bool] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> Generator[str, None, None]:
        """Generate text token-by-token yielding string fragments incrementally.

        Args:
            prompt: Text prompt input string.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            do_sample: Explicit boolean to force sampling (True) or greedy decoding (False).
                       If None: defaults to False (greedy decoding) unless temperature != 1.0.
            stop_token_ids: Optional list of token IDs that stop generation when produced.

        Yields:
            Decoded string tokens/fragments as they are generated.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("ApahRuntime is uninitialized or has been unloaded.")

        if do_sample is None:
            # Default to greedy decoding as required unless non-default temperature is supplied
            do_sample = (temperature != 1.0) and (temperature > 0.0)

        # Build set of stop token IDs
        stop_set = set(stop_token_ids) if stop_token_ids is not None else set()
        if self.tokenizer.eos_token_id is not None:
            stop_set.add(self.tokenizer.eos_token_id)

        # Reset GPU peak memory stats if on CUDA
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        start_time = time.perf_counter()

        # Tokenize input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        seq_len = input_ids.shape[1]

        attention_mask = (
            inputs.attention_mask.to(self.device)
            if "attention_mask" in inputs
            else torch.ones((1, seq_len), dtype=torch.long, device=self.device)
        )

        # Prefill phase (prompt forward pass with manual KV cache enable)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )

        logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)
        past_key_values = outputs.past_key_values

        # Generate first token & calculate TTFT
        next_token_id = self._sample_next_token(logits, temperature, do_sample)
        ttft_sec = time.perf_counter() - start_time

        generated_token_ids: List[int] = []
        token_val = int(next_token_id.item())

        if token_val in stop_set:
            self._record_metrics(start_time, ttft_sec, generated_token_ids)
            return

        generated_token_ids.append(token_val)
        all_tokens = generated_token_ids.copy()

        # Yield first decoded fragment
        first_text = self.tokenizer.decode(token_val, skip_special_tokens=True)
        if first_text:
            yield first_text

        prev_decoded_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)

        # Token-by-token decode loop with manual KV cache updates
        for step in range(1, max_new_tokens):
            current_seq_len = seq_len + step
            attention_mask = torch.ones((1, current_seq_len), dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token_id,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            next_token_id = self._sample_next_token(logits, temperature, do_sample)
            token_val = int(next_token_id.item())

            if token_val in stop_set:
                break

            generated_token_ids.append(token_val)
            all_tokens.append(token_val)

            current_decoded_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
            chunk = current_decoded_text[len(prev_decoded_text) :]
            prev_decoded_text = current_decoded_text

            if chunk:
                yield chunk

        self._record_metrics(start_time, ttft_sec, generated_token_ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: Optional[bool] = None,
        stop_token_ids: Optional[List[int]] = None,
    ) -> str:
        """Generate text for a single prompt using manual KV cache.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            do_sample: Explicit boolean to force sampling (True) or greedy decoding (False).
            stop_token_ids: Optional list of token IDs to stop generation.

        Returns:
            Complete generated output text.
        """
        chunks = list(
            self.generate_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                stop_token_ids=stop_token_ids,
            )
        )
        return "".join(chunks)

    def _record_metrics(
        self,
        start_time: float,
        ttft_sec: float,
        generated_token_ids: List[int],
    ) -> None:
        """Record and log runtime metrics."""
        total_time_sec = time.perf_counter() - start_time
        gen_count = len(generated_token_ids)

        if gen_count > 1:
            decode_time = total_time_sec - ttft_sec
            tokens_per_sec = (gen_count - 1) / decode_time if decode_time > 0 else (gen_count / total_time_sec)
        elif gen_count == 1:
            tokens_per_sec = 1.0 / total_time_sec if total_time_sec > 0 else 0.0
        else:
            tokens_per_sec = 0.0

        peak_gpu_mb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)

        self.last_metrics = GenerationMetrics(
            ttft_sec=ttft_sec,
            total_time_sec=total_time_sec,
            tokens_per_sec=tokens_per_sec,
            generated_tokens=gen_count,
            peak_gpu_memory_mb=peak_gpu_mb,
        )

        logger.info(
            f"Generation complete: TTFT={ttft_sec * 1000:.2f} ms | "
            f"Speed={tokens_per_sec:.2f} tok/s | "
            f"Generated={gen_count} tokens | "
            f"Peak GPU Mem={peak_gpu_mb:.2f} MB"
        )

    def unload(self) -> None:
        """Unload model from GPU memory and clear CUDA memory cache."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        """Clean up resources on garbage collection."""
        try:
            self.unload()
        except Exception:
            pass
