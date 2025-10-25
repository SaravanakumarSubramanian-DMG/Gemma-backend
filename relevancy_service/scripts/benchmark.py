import argparse
import base64
import json
import time
from pathlib import Path

import httpx


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--endpoint", default="http://localhost:8080/relevancy")
	parser.add_argument("--images_dir", required=True)
	parser.add_argument("--text_file", required=True)
	parser.add_argument("--repeat", type=int, default=5)
	args = parser.parse_args()

	text = Path(args.text_file).read_text().strip()
	images = sorted(list(Path(args.images_dir).glob("*.jpg"))) + sorted(list(Path(args.images_dir).glob("*.png")))
	if not images:
		raise SystemExit("No images found")

	latencies = []
	with httpx.Client(timeout=60) as client:
		for i in range(args.repeat):
			for img_path in images:
				img_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
				payload = {"text": text, "image_base64": img_b64}
				start = time.time()
				resp = client.post(args.endpoint, json=payload)
				resp.raise_for_status()
				latencies.append((time.time() - start) * 1000.0)

	print(json.dumps({
		"count": len(latencies),
		"p50_ms": float(sorted(latencies)[len(latencies)//2]),
		"p95_ms": float(sorted(latencies)[int(len(latencies)*0.95)]),
		"avg_ms": float(sum(latencies) / len(latencies)),
	}, indent=2))
