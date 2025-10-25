import argparse
import subprocess
from pathlib import Path


def run(cmd):
	print("$", " ".join(cmd))
	subprocess.check_call(cmd)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--onnx", required=True)
	parser.add_argument("--engine", required=True)
	parser.add_argument("--fp16", action="store_true", default=True)
	parser.add_argument("--int8", action="store_true", default=False)
	parser.add_argument("--workspace", type=int, default=4096)
	args = parser.parse_args()

	onnx_path = Path(args.onnx).resolve()
	engine_path = Path(args.engine).resolve()
	cmd = [
		"trtexec",
		f"--onnx={str(onnx_path)}",
		f"--saveEngine={str(engine_path)}",
		f"--workspace={args.workspace}",
		"--verbose",
	]
	if args.fp16:
		cmd.append("--fp16")
	if args.int8:
		cmd.append("--int8")

	run(cmd)
