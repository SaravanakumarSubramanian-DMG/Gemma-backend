import argparse
import subprocess
from pathlib import Path


def run(cmd):
	print("$", " ".join(cmd))
	subprocess.check_call(cmd)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--onnx", required=True)
	parser.add_argument("--outdir", required=True)
	parser.add_argument("--fp16", action="store_true", default=True)
	args = parser.parse_args()

	onnx_path = Path(args.onnx).resolve()
	outdir = Path(args.outdir).resolve()
	outdir.mkdir(parents=True, exist_ok=True)

	cmd = [
		"onnx2tf",
		"-i",
		str(onnx_path),
		"-o",
		str(outdir),
	]
	if args.fp16:
		cmd.extend(["-qt", "f16"])

	run(cmd)
