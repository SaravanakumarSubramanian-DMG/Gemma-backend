import os
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


@runtime_checkable
class EmbeddingModel(Protocol):
    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_image(self, image: Image.Image) -> np.ndarray: ...


class SiglipEmbeddingModel:
    """Generic HF image-text embedding model wrapper using AutoModel.

    Works with SigLIP, SigLIP2, CLIP-like encoders that expose
    get_text_features/get_image_features; otherwise falls back to pooling CLS.
    """

    def __init__(self, model_id: str | None = None, token: str | None = None) -> None:
        self.model_id = model_id or os.environ.get("EMBED_MODEL_ID", "google/siglip2-large-patch16-384")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.model_id, token=token)
        self._model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            token=token,
        )
        self._model.eval()
        self._model.to(self.device)

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        # Batch dimension of 1 for consistent handling
        inputs = self._processor(text=[text], padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if hasattr(self._model, "get_text_features"):
            features = self._model.get_text_features(**inputs)
        else:
            outputs = self._model(**inputs)
            features = outputs.last_hidden_state[:, 0]
        vec = features[0].detach().float().cpu().numpy()
        return self._l2_normalize(vec)

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if hasattr(self._model, "get_image_features"):
            features = self._model.get_image_features(**inputs)
        else:
            outputs = self._model(**inputs)
            features = outputs.last_hidden_state[:, 0]
        vec = features[0].detach().float().cpu().numpy()
        return self._l2_normalize(vec)

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        denom = float(np.linalg.norm(vec)) or 1.0
        return (vec / denom).astype(np.float32)


class CosineSimilarityCalculator:
    """Pure function object to compute cosine similarity between two L2 vectors."""

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        # Both expected L2-normalized. Guard against shape or NaNs.
        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("cosine expects 1D vectors")
        if a.shape[0] != b.shape[0]:
            raise ValueError("mismatched vector sizes")
        val = float(np.dot(a, b))
        if not np.isfinite(val):
            return float("nan")
        return max(-1.0, min(1.0, val))


class EmbeddingService:
    """Facade for embedding + similarity.

    Responsibilities:
    - Expose stable API to obtain embeddings and compute similarities
    - Hide model specifics behind `EmbeddingModel`
    """

    def __init__(self, model: EmbeddingModel, similarity: CosineSimilarityCalculator) -> None:
        self._model = model
        self._similarity = similarity

    def embed_text(self, text: str) -> np.ndarray:
        return self._model.embed_text(text)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self._model.embed_image(image)

    def cosine_similarity(self, image_vec: np.ndarray, text_vec: np.ndarray) -> float:
        return self._similarity.compute(image_vec, text_vec)


_SERVICE_SINGLETON: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        token = os.environ.get("HF_TOKEN", None)
        model = SiglipEmbeddingModel(token=token)
        sim = CosineSimilarityCalculator()
        _SERVICE_SINGLETON = EmbeddingService(model=model, similarity=sim)
    return _SERVICE_SINGLETON


