import hashlib
from typing import List


def normalize_text(text: str) -> str:
	return " ".join(text.strip().lower().split())


def hash_text(text: str) -> str:
	norm = normalize_text(text)
	return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def batch_hash_text(texts: List[str]) -> List[str]:
	return [hash_text(t) for t in texts]
