from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

from .base import EncoderInterface
from ..config import settings


class SiglipClipEncoder(EncoderInterface):
	def __init__(self, model_name: str, device: str = "cuda", image_size: int = 384, hf_token: Optional[str] = None) -> None:
		self._name = model_name
		self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
		self.processor = AutoProcessor.from_pretrained(model_name, use_auth_token=hf_token or settings.hf_token)
		self.model = AutoModel.from_pretrained(
			model_name,
			torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
			use_auth_token=hf_token or settings.hf_token,
		)
		self.model.to(self.device)
		self.model.eval()
		self.image_size = image_size

	@property
	def name(self) -> str:
		return self._name

	@torch.inference_mode()
	def encode_text(self, texts: List[str]) -> List[List[float]]:
		inputs = self.processor(text=texts, padding=True, truncation=True, return_tensors="pt")
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		if hasattr(self.model, "get_text_features"):
			emb = self.model.get_text_features(**inputs)
		else:
			outputs = self.model(**inputs)
			emb = outputs.last_hidden_state[:, 0]
		emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
		return emb.float().cpu().tolist()

	@torch.inference_mode()
	def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
		inputs = self.processor(images=images, return_tensors="pt")
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		if hasattr(self.model, "get_image_features"):
			emb = self.model.get_image_features(**inputs)
		else:
			outputs = self.model(**inputs)
			emb = outputs.last_hidden_state[:, 0]
		emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
		return emb.float().cpu().tolist()
