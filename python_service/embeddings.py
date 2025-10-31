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
            last_hidden = outputs.last_hidden_state  # [B, T, D]
            mask = inputs.get("attention_mask", None)
            if mask is None:
                features = last_hidden[:, 0]
            else:
                mask = mask.unsqueeze(-1)  # [B, T, 1]
                features = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
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

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        inputs = self._processor(text=texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if hasattr(self._model, "get_text_features"):
            feats = self._model.get_text_features(**inputs)
        else:
            outputs = self._model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs.get("attention_mask", None)
            if mask is None:
                feats = last_hidden[:, 0]
            else:
                mask = mask.unsqueeze(-1)
                feats = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        vecs = feats.detach().float().cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vecs / norms).astype(np.float32)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=[im.convert("RGB") for im in images], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if hasattr(self._model, "get_image_features"):
            feats = self._model.get_image_features(**inputs)
        else:
            outputs = self._model(**inputs)
            feats = outputs.last_hidden_state[:, 0]
        vecs = feats.detach().float().cpu().numpy()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vecs / norms).astype(np.float32)

    def get_logit_scale(self) -> float | None:
        scale = getattr(self._model, "logit_scale", None)
        if scale is None:
            return None
        try:
            if isinstance(scale, torch.Tensor):
                return float(scale.item())
            return float(scale)
        except Exception:
            return None

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

    def compute_matrix(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute all-pairs cosine for two batches of L2-normalized vectors.

        Args:
            a: shape [N, D]
            b: shape [M, D]
        Returns:
            sims: shape [N, M]
        """
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("cosine matrix expects 2D arrays")
        if a.shape[1] != b.shape[1]:
            raise ValueError("mismatched vector sizes")
        sims = a @ b.T
        sims = np.clip(sims, -1.0, 1.0)
        return sims.astype(np.float32)

    def apply_temperature(self, sims: np.ndarray, logit_scale: float | None) -> np.ndarray:
        """Optionally apply CLIP-style temperature scaling to similarity scores."""
        if logit_scale is None:
            return sims
        return (sims * float(np.exp(logit_scale))).astype(np.float32)


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

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if hasattr(self._model, "embed_texts"):
            return getattr(self._model, "embed_texts")(texts)  # type: ignore[misc]
        return np.stack([self._model.embed_text(t) for t in texts], axis=0)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        if hasattr(self._model, "embed_images"):
            return getattr(self._model, "embed_images")(images)  # type: ignore[misc]
        return np.stack([self._model.embed_image(im) for im in images], axis=0)

    def cosine_similarity_matrix(self, image_vecs: np.ndarray, text_vecs: np.ndarray) -> np.ndarray:
        return self._similarity.compute_matrix(image_vecs, text_vecs)

    def build_stage_prompts(self, description: str) -> list[str]:
        return [
            f"A BEFORE photo just prior to work starting. Context: {description}",
            f"An IN-PROGRESS photo while work is being done. Context: {description}",
            f"An AFTER photo showing completed work and final result. Context: {description}",
        ]

    def stage_text_embeddings(self, description: str) -> np.ndarray:
        prompts = self.build_stage_prompts(description)
        return self.embed_texts(prompts)

    def stage_probabilities_for_images(self, image_vecs: np.ndarray, stage_text_vecs: np.ndarray) -> np.ndarray:
        """Return per-image probabilities over stages using softmax over cosine sims.

        Args:
            image_vecs: [N, D] L2-normalized image embeddings
            stage_text_vecs: [3, D] L2-normalized text embeddings for stages
        Returns:
            probs: [N, 3] probabilities across stages per image
        """
        if image_vecs.ndim != 2 or stage_text_vecs.ndim != 2:
            raise ValueError("expected 2D arrays")
        sims = image_vecs @ stage_text_vecs.T  # [N, 3]
        sims = sims - sims.max(axis=1, keepdims=True)
        exps = np.exp(sims)
        probs = exps / np.clip(exps.sum(axis=1, keepdims=True), 1e-9, None)
        return probs.astype(np.float32)

    def scores_against_description(self, image_vecs: np.ndarray, description_vec: np.ndarray) -> np.ndarray:
        if description_vec.ndim != 1:
            raise ValueError("description_vec must be 1D")
        return (image_vecs @ description_vec).astype(np.float32)


_SERVICE_SINGLETON: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        token = os.environ.get("HF_TOKEN", None)
        model = SiglipEmbeddingModel(token=token)
        sim = CosineSimilarityCalculator()
        _SERVICE_SINGLETON = EmbeddingService(model=model, similarity=sim)
    return _SERVICE_SINGLETON


