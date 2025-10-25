import base64
import hashlib
import io
import os
from typing import Any, List, Optional

import httpx
from PIL import Image


def load_image_from_base64(img_b64: str) -> Image.Image:
	if "," in img_b64:
		img_b64 = img_b64.split(",", 1)[1]
	data = base64.b64decode(img_b64)
	return Image.open(io.BytesIO(data)).convert("RGB")


def load_image_from_url(url: str, timeout: float = 10.0) -> Image.Image:
	with httpx.Client(timeout=timeout, follow_redirects=True) as client:
		resp = client.get(url)
		resp.raise_for_status()
		return Image.open(io.BytesIO(resp.content)).convert("RGB")


def load_image_from_path(path: str) -> Image.Image:
	with open(path, "rb") as f:
		return Image.open(io.BytesIO(f.read())).convert("RGB")


def load_image_from_any(val: Any) -> Image.Image:
	"""Accepts dict with keys b64/url/path or a string (url, path, or base64)."""
	if isinstance(val, dict):
		b64 = val.get("b64") or val.get("image_base64")
		url = val.get("url") or val.get("image_url")
		path = val.get("path")
		if b64:
			return load_image_from_base64(b64)
		if url:
			return load_image_from_url(url)
		if path:
			return load_image_from_path(path)
		raise ValueError("Unsupported image object; expected b64/url/path")
	if isinstance(val, Image.Image):
		return val.convert("RGB")
	if isinstance(val, str):
		s = val.strip()
		if s.startswith("http://") or s.startswith("https://"):
			return load_image_from_url(s)
		if os.path.exists(s):
			return load_image_from_path(s)
		# fallback assume base64
		return load_image_from_base64(s)
	raise ValueError("Unsupported image value type")


def hash_image_bytes(image: Image.Image, format: str = "PNG") -> str:
	buf = io.BytesIO()
	image.save(buf, format=format)
	return hashlib.sha256(buf.getvalue()).hexdigest()


def load_images_from_base64_list(images_b64: List[str]) -> List[Image.Image]:
	return [load_image_from_base64(b) for b in images_b64]


def load_images_from_url_list(images_url: List[str]) -> List[Image.Image]:
	return [load_image_from_url(u) for u in images_url]
