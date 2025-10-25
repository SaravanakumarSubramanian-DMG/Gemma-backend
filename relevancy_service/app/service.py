import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import settings
from .models.base import EncoderInterface
from .models.siglip_clip import SiglipClipEncoder
from .models.gemma_optional import GemmaVLMScorer
from .utils.cache import LRUCache
from .utils.text import hash_text
from .utils.image import hash_image_bytes, load_image_from_any
from .utils.similarity import cosine_similarity_np, to_unit_interval

logger = logging.getLogger("relevancy_service")
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))


class ModelLoader:
	@staticmethod
	def load(model_name: str, device: str, image_size: int, hf_token: Optional[str]) -> EncoderInterface:
		return SiglipClipEncoder(model_name=model_name, device=device, image_size=image_size, hf_token=hf_token)


class RelevancyService:
	def __init__(self, encoder: Optional[EncoderInterface] = None, hf_token: Optional[str] = None) -> None:
		self.encoder = encoder or ModelLoader.load(settings.model_name, settings.device, settings.image_size, hf_token or settings.hf_token)
		self.text_cache = LRUCache[str, List[float]](capacity=settings.embed_cache_max_items)
		self.image_cache = LRUCache[str, List[float]](capacity=settings.embed_cache_max_items)
		self.gemma = GemmaVLMScorer(settings.gemma_model_name, settings.device, hf_token or settings.hf_token) if (settings.use_gemma_explanations or settings.primary_scorer == "gemma") else None

	@property
	def model_name(self) -> str:
		return self.encoder.name

	def encode_texts(self, texts: List[str]) -> List[List[float]]:
		keys = [hash_text(t) for t in texts]
		results: List[Optional[List[float]]] = [self.text_cache.get(k) for k in keys]
		missing_idx = [i for i, r in enumerate(results) if r is None]
		if missing_idx:
			missing_texts = [texts[i] for i in missing_idx]
			start = time.time()
			encoded = self.encoder.encode_text(missing_texts)
			logger.debug("encode_text batch=%d time_ms=%.2f", len(missing_texts), (time.time() - start) * 1000)
			for idx, emb in zip(missing_idx, encoded):
				results[idx] = emb
				self.text_cache.put(keys[idx], emb)
		return [r for r in results if r is not None]  # type: ignore

	def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
		keys = [hash_image_bytes(img) for img in images]
		results: List[Optional[List[float]]] = [self.image_cache.get(k) for k in keys]
		missing_idx = [i for i, r in enumerate(results) if r is None]
		if missing_idx:
			missing_imgs = [images[i] for i in missing_idx]
			start = time.time()
			encoded = self.encoder.encode_images(missing_imgs)
			logger.debug("encode_image batch=%d time_ms=%.2f", len(missing_imgs), (time.time() - start) * 1000)
			for idx, emb in zip(missing_idx, encoded):
				results[idx] = emb
				self.image_cache.put(keys[idx], emb)
		return [r for r in results if r is not None]  # type: ignore

	def score_pairs_dual(self, texts: List[str], images: List[Image.Image]) -> List[float]:
		text_emb = np.array(self.encode_texts(texts), dtype=np.float32)
		img_emb = np.array(self.encode_images(images), dtype=np.float32)
		assert text_emb.shape[0] == img_emb.shape[0]
		scores: List[float] = []
		for i in range(text_emb.shape[0]):
			cos = cosine_similarity_np(text_emb[i], img_emb[i])
			scores.append(to_unit_interval(float(cos)))
		return scores

	def score_pairs(self, texts: List[str], images: List[Image.Image]) -> List[float]:
		start = time.time()
		if settings.primary_scorer == "gemma" and self.gemma is not None:
			try:
				results = self.gemma.score_with_reason(images, texts)
				scores = [float(r.get("score", 0.5)) for r in results]
				logger.info("score_pairs_gemma batch=%d time_ms=%.2f", len(texts), (time.time() - start) * 1000)
				return scores
			except Exception as e:
				logger.warning("gemma_primary_failed_fallback_dual: %s", e)
		# fallback or primary dual
		scores = self.score_pairs_dual(texts, images)
		logger.info("score_pairs_dual batch=%d time_ms=%.2f", len(texts), (time.time() - start) * 1000)
		return scores

	def score_single_with_meta(self, text: str, image: Image.Image, image_url: Optional[str] = None) -> dict:
		start = time.time()
		score = self.score_pairs([text], [image])[0]
		inference_ms = (time.time() - start) * 1000.0
		vision_summary = None
		reason = None
		if self.gemma is not None and settings.primary_scorer != "gemma":
			try:
				g = self.gemma.score_with_reason([image], [text])[0]
				vision_summary = g.get("reason")
				reason = g.get("reason")
			except Exception as e:
				logger.warning("gemma_explain_failed: %s", e)
		meta = {
			"inference_ms": float(inference_ms),
			"text": text,
			"image_url": image_url,
			"model": self.model_name if settings.primary_scorer == "dual" else settings.gemma_model_name,
			"vision_summary": vision_summary,
			"relevancy_reason": reason,
		}
		logger.info("inference score=%.3f ms=%.2f text_len=%d image_url=%s scorer=%s", score, inference_ms, len(text), str(image_url), settings.primary_scorer)
		return {"score": float(score), "meta": meta}

	def group_score_and_analysis(
		self,
		items: List[Any],
		description: str,
		summary: Optional[str],
		stage_label: str,
		skip_desc: bool,
	) -> List[Dict[str, Any]]:
		results: List[Dict[str, Any]] = []
		if not items:
			return results
		# Load all images first
		loaded_images: List[Optional[Image.Image]] = []
		for val in (items or []):
			try:
				loaded_images.append(load_image_from_any(val))
			except Exception as e:
				logger.warning("group_item_load_error stage=%s err=%s", stage_label, e)
				loaded_images.append(None)

		# Batch score using dual or gemma primary
		batch_size = max(1, int(settings.batch_size))
		scores_percent: List[Optional[float]] = [None] * len(loaded_images)
		start_all = time.time()
		for i in range(0, len(loaded_images), batch_size):
			batch_imgs = [img for img in loaded_images[i : i + batch_size] if img is not None]
			idxs = [j for j in range(i, min(i + batch_size, len(loaded_images))) if loaded_images[j] is not None]
			if not batch_imgs:
				continue
			texts = [description] * len(batch_imgs)
			batch_scores = self.score_pairs(texts, batch_imgs)
			for j, s in zip(idxs, batch_scores):
				scores_percent[j] = float(s) * 100.0
		logger.info("group_score stage=%s count=%d time_ms=%.2f", stage_label, len(loaded_images), (time.time() - start_all) * 1000)

		# Optional Gemma explanations in batches
		vision_summaries: List[Optional[str]] = [None] * len(loaded_images)
		analyses: List[Optional[Dict[str, Any]]] = [None] * len(loaded_images)
		raws: List[Optional[Dict[str, Any]]] = [None] * len(loaded_images)
		if self.gemma is not None:
			for i in range(0, len(loaded_images), batch_size):
				batch_imgs = [img for img in loaded_images[i : i + batch_size] if img is not None]
				idxs = [j for j in range(i, min(i + batch_size, len(loaded_images))) if loaded_images[j] is not None]
				if not batch_imgs:
					continue
				try:
					# Explanations against description
					g_expl_list = self.gemma.score_with_reason(batch_imgs, [description] * len(batch_imgs))
					# Optional short vision summary
					g_desc_list = self.gemma.score_with_reason(batch_imgs, ["Describe briefly"] * len(batch_imgs)) if not skip_desc else [{"reason": None}] * len(batch_imgs)
					for k, j in enumerate(idxs):
						analyses[j] = {"explanation": g_expl_list[k].get("reason"), "verdict": ""}
						vision_summaries[j] = g_desc_list[k].get("reason") if not skip_desc else None
						raws[j] = {"analysis": g_expl_list[k]}
				except Exception as e:
					logger.warning("gemma_batch_explain_failed stage=%s err=%s", stage_label, e)

		# Assemble results
		for idx in range(len(loaded_images)):
			if loaded_images[idx] is None:
				results.append({"error": "invalid image"})
				continue
			record: Dict[str, Any] = {
				"text_relevancy": scores_percent[idx] if scores_percent[idx] is not None else float("nan"),
				"vision_summary": vision_summaries[idx],
				"analysis": analyses[idx],
				"_raw": raws[idx] if raws[idx] is not None else {},
			}
			results.append(record)
		return results

	def pairwise_similarity_matrix(
		self,
		a_list: List[Any],
		b_list: List[Any],
	) -> Tuple[List[List[float]], List[List[Dict[str, Any]]]]:
		A_imgs = []
		B_imgs = []
		for val in a_list or []:
			try:
				A_imgs.append(load_image_from_any(val))
			except Exception:
				A_imgs.append(None)
		for val in b_list or []:
			try:
				B_imgs.append(load_image_from_any(val))
			except Exception:
				B_imgs.append(None)
		A_valid = [img for img in A_imgs if img is not None]
		B_valid = [img for img in B_imgs if img is not None]
		if not A_valid or not B_valid:
			return [], []
		A_emb = np.array(self.encode_images(A_valid), dtype=np.float32)
		B_emb = np.array(self.encode_images(B_valid), dtype=np.float32)
		matrix: List[List[float]] = []
		detailed: List[List[Dict[str, Any]]] = []
		ai = 0
		for a_img in A_imgs:
			row: List[float] = []
			drow: List[Dict[str, Any]] = []
			if a_img is None:
				for _ in B_imgs:
					row.append(float("nan"))
					drow.append({"score": float("nan"), "explanation": "invalid image"})
				matrix.append(row)
				detailed.append(drow)
				continue
			bi = 0
			for b_img in B_imgs:
				if b_img is None:
					row.append(float("nan"))
					drow.append({"score": float("nan"), "explanation": "invalid image"})
					bi += 1
					continue
				cos = float(np.dot(A_emb[ai], B_emb[bi]) / (np.linalg.norm(A_emb[ai]) * np.linalg.norm(B_emb[bi]) + 1e-8))
				score01 = to_unit_interval(cos)
				row.append(score01 * 100.0)
				drow.append({"score": score01 * 100.0})
				bi += 1
			matrix.append(row)
			detailed.append(drow)
			ai += 1
		return matrix, detailed
