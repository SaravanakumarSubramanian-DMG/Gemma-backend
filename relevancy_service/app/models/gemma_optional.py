from typing import List, Optional

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
try:
	from transformers import Gemma3nForConditionalGeneration
except Exception:  # older transformers
	Gemma3nForConditionalGeneration = None  # type: ignore
from PIL import Image

from ..config import settings


class GemmaVLMScorer:
	"""Optional VLM cross-modal scorer: returns a scalar relevancy score [0,1] and brief justification."""

	def __init__(self, model_name: Optional[str] = None, device: str = "cuda", hf_token: Optional[str] = None) -> None:
		self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
		model_id = model_name or settings.gemma_model_name
		token = hf_token or settings.hf_token
		self._is_gemma3n = "gemma-3n" in (model_id or "").lower() and Gemma3nForConditionalGeneration is not None
		self.processor = AutoProcessor.from_pretrained(model_id, token=token)
		if self._is_gemma3n:
			self.model = Gemma3nForConditionalGeneration.from_pretrained(
				model_id,
				device_map="auto" if self.device.type == "cuda" else None,
				torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
				token=token,
			)
			# Ensure attention uses a safe implementation without optional extensions
			if hasattr(self.model.config, "_attn_implementation"):
				self.model.config._attn_implementation = "eager"
		else:
			self.model = AutoModelForVision2Seq.from_pretrained(
				model_id,
				torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
				token=token,
			)
		self.model.to(self.device)
		self.model.eval()

	@torch.inference_mode()
	def score_with_reason(self, images: List[Image.Image], texts: List[str]) -> List[dict]:
		if self._is_gemma3n:
			# Use chat template with multimodal messages
			messages_batch = []
			for img, t in zip(images, texts):
				messages_batch.append([
					{"role": "system", "content": [{"type": "text", "text": "You are a strict inspector."}]},
					{
						"role": "user",
						"content": [
							{"type": "image", "image": img},
							{"type": "text", "text": (
								"For the job description and image, output one line: a relevance score between 0 and 1, then a short reason. Format: SCORE | REASON\nJob: "
								+ t
							)},
						],
					},
				])
			encoded_list = [
				self.processor.apply_chat_template(
					m,
					add_generation_prompt=True,
					tokenize=True,
					return_dict=True,
					return_tensors="pt",
				).to(self.device)
				for m in messages_batch
			]
			results: List[dict] = []
			for inputs in encoded_list:
				input_len = inputs["input_ids"].shape[-1]
				gen = self.model.generate(**inputs, max_new_tokens=64, do_sample=False)
				gen = gen[0][input_len:]
				decoded = self.processor.decode(gen, skip_special_tokens=True)
				try:
					parts = decoded.strip().split("|")
					score = float(parts[0].strip().split()[0])
					score = max(0.0, min(1.0, score))
					reason = parts[1].strip() if len(parts) > 1 else ""
				except Exception:
					score, reason = 0.5, "uncertain"
				results.append({"score": score, "reason": reason})
			return results
		# Generic VLM path
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
