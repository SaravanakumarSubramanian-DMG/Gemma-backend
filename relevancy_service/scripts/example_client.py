import argparse
import base64
import json
from pathlib import Path

import httpx


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--endpoint", default="http://localhost:8080/relevancy")
	parser.add_argument("--image", required=True)
	parser.add_argument("--text", required=True)
	args = parser.parse_args()

	img_b64 = base64.b64encode(Path(args.image).read_bytes()).decode("utf-8")
	payload = {"text": args.text, "image_base64": img_b64}
	with httpx.Client(timeout=30) as client:
		resp = client.post(args.endpoint, json=payload)
		resp.raise_for_status()
		print(json.dumps(resp.json(), indent=2))
