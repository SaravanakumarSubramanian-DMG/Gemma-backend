import argparse
from typing import Dict

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor


class TextWrapper(torch.nn.Module):
	def __init__(self, model: torch.nn.Module) -> None:
		super().__init__()
		self.model = model

	def forward(self, **inputs):
		if hasattr(self.model, "get_text_features"):
			return self.model.get_text_features(**inputs)
		out = self.model(**inputs)
		return out.last_hidden_state[:, 0]


class ImageWrapper(torch.nn.Module):
	def __init__(self, model: torch.nn.Module) -> None:
		super().__init__()
		self.model = model

	def forward(self, **inputs):
		if hasattr(self.model, "get_image_features"):
			return self.model.get_image_features(**inputs)
		out = self.model(**inputs)
		return out.last_hidden_state[:, 0]


def export_text(model_name: str, output: str, device: str = "cpu") -> None:
	processor = AutoProcessor.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name).to(device)
	model.eval()
	wrap = TextWrapper(model)
	dummy = processor(text=["hello world"], return_tensors="pt")
	dummy = {k: v.to(device) for k, v in dummy.items()}
	symbolic_names = {0: "batch", 1: "hidden"}
	torch.onnx.export(
		wrap,
		tuple(),
		output,
		export_params=True,
		do_constant_folding=True,
		input_names=list(dummy.keys()),
		output_names=["text_features"],
		dynamic_axes={**{k: {0: "batch", 1: "seq"} for k in dummy.keys()}, "text_features": symbolic_names},
		opset_version=17,
		kwargs=dummy,  # pass inputs as kwargs
	)


def export_image(model_name: str, output: str, device: str = "cpu", image_size: int = 384) -> None:
	processor = AutoProcessor.from_pretrained(model_name)
	model = AutoModel.from_pretrained(model_name).to(device)
	model.eval()
	wrap = ImageWrapper(model)
	dummy_img = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
	dummy = processor(images=[dummy_img], return_tensors="pt")
	dummy = {k: v.to(device) for k, v in dummy.items()}
	symbolic_names = {0: "batch", 1: "hidden"}
	torch.onnx.export(
		wrap,
		tuple(),
		output,
		export_params=True,
		do_constant_folding=True,
		input_names=list(dummy.keys()),
		output_names=["image_features"],
		dynamic_axes={**{k: {0: "batch"} for k in dummy.keys()}, "image_features": symbolic_names},
		opset_version=17,
		kwargs=dummy,
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default="google/siglip-so400m-patch14-384")
	parser.add_argument("--text_out", default="text_encoder.onnx")
	parser.add_argument("--image_out", default="image_encoder.onnx")
	parser.add_argument("--device", default="cpu")
	parser.add_argument("--image_size", type=int, default=384)
	args = parser.parse_args()
	export_text(args.model, args.text_out, args.device)
	export_image(args.model, args.image_out, args.device, args.image_size)
