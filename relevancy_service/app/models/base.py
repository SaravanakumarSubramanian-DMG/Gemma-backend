from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image


class EncoderInterface(ABC):
	@abstractmethod
	def encode_text(self, texts: List[str]) -> List[List[float]]:
		...

	@abstractmethod
	def encode_images(self, images: List[Image.Image]) -> List[List[float]]:
		...

	@property
	@abstractmethod
	def name(self) -> str:
		...
