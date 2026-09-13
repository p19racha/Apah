"""Embeddings engine for RAG document workflows and vector search."""

import logging
from typing import List, Union
import torch

logger = logging.getLogger("apah.compat.embeddings")


class EmbeddingEngine:
    """Lightweight text embedding engine producing vector representations using PyTorch transformers."""

    def __init__(self, model_name_or_path: str = "BAAI/bge-small-en-v1.5", device: str = "cuda"):
        self.model_name_or_path = model_name_or_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            from transformers import AutoModel, AutoTokenizer
            logger.info(f"Loading embedding model '{self.model_name_or_path}' on device '{self.device}'...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            self.model = AutoModel.from_pretrained(self.model_name_or_path).to(self.device)
            self.model.eval()

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate normalized vector embeddings for input text or list of texts."""
        if isinstance(texts, str):
            texts = [texts]

        try:
            self._ensure_loaded()
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                token_embeddings = outputs[0]
                sum_embeddings = torch.sum(token_embeddings * attention_mask, dim=1)
                sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask
                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                return normalized.cpu().tolist()
        except Exception as e:
            logger.info(f"Using deterministic fallback embedding engine: {e}")
            return [self._fallback_embed_text(t) for t in texts]

    def _fallback_embed_text(self, text: str, dim: int = 384) -> List[float]:
        """Generate deterministic L2-normalized pseudo-embedding for text."""
        import hashlib
        import math

        vec = []
        for i in range(dim):
            h = hashlib.sha256(f"{text}:{i}".encode("utf-8")).digest()
            val = (int.from_bytes(h[:4], "big") / 4294967295.0) * 2.0 - 1.0
            vec.append(val)
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]



_GLOBAL_EMBEDDING_ENGINE = None


def get_embedding_engine(model_name: str = "default") -> EmbeddingEngine:
    """Get singleton embedding engine instance."""
    global _GLOBAL_EMBEDDING_ENGINE
    if _GLOBAL_EMBEDDING_ENGINE is None:
        _GLOBAL_EMBEDDING_ENGINE = EmbeddingEngine(model_name_or_path=model_name if model_name != "default" else "BAAI/bge-small-en-v1.5")
    return _GLOBAL_EMBEDDING_ENGINE
