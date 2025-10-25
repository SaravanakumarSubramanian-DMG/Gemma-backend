from typing import List, Optional

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

from ..config import settings


class GemmaVLMScorer:
	"""Optional VLM cross-modal scorer: returns a scalar relevancy score [0,1] and brief justification."""

	def __init__(self, model_name: Optional[str] = None, device: str = "cuda", hf_token: Optional[str] = None) -> None:
		self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
		model_id = model_name or settings.gemma_model_name
		token = hf_token or settings.hf_token
		self.processor = AutoProcessor.from_pretrained(model_id, use_auth_token=token)
		self.model = AutoModelForVision2Seq.from_pretrained(
			model_id,
			torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
			use_auth_token=token,
		)
		self.model.to(self.device)
		self.model.eval()

	@torch.inference_mode()
	def score_with_reason(self, images: List[Image.Image], texts: List[str]) -> List[dict]:
		prompts = [
			(
				"You are a strict inspector. For the given job description and image, output on one line: "
				"a relevance score between 0 and 1, then a short reason. Format: SCORE | REASON\n"
				"Job: " + t
			)
			for t in texts
		]
		inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		outputs = self.model.generate(**inputs, max_new_tokens=32)
		decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
		results = []
		for s in decoded:
			try:
				parts = s.strip().split("|")
				score = float(parts[0].strip().split()[0])
				score = max(0.0, min(1.0, score))
				reason = parts[1].strip() if len(parts) > 1 else ""
			except Exception:
				score, reason = 0.5, "uncertain"
			results.append({"score": score, "reason": reason})
		return results
