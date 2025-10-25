from typing import List
import numpy as np
import torch


def l2_normalize_np(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
	norm = np.linalg.norm(v, axis=-1, keepdims=True)
	return v / np.clip(norm, eps, None)


def l2_normalize_torch(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	norm = torch.linalg.norm(v, dim=-1, keepdim=True)
	return v / torch.clamp(norm, min=eps)


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
	a_n = l2_normalize_np(a)
	b_n = l2_normalize_np(b)
	return float(np.sum(a_n * b_n, axis=-1))


def cosine_similarity_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
	a_n = l2_normalize_torch(a)
	b_n = l2_normalize_torch(b)
	return torch.sum(a_n * b_n, dim=-1)


def to_unit_interval(sim: float) -> float:
	# Map cosine [-1, 1] to [0, 1]
	return (sim + 1.0) / 2.0
