import os
import base64
import io
from typing import List, Dict, Any
import math
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from pydantic import BaseModel
from PIL import Image
from .exif_utils import extract_exif_fields
import torch
from transformers import AutoProcessor, Gemma3nForConditionalGeneration
from huggingface_hub import login as hf_login
import requests
import json

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = "google/gemma-3n-E4B-it"

import time
import logging

app = FastAPI()

# Basic structured logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def _clean_chat_output(text: str) -> str:
    """Remove chat role markers and prompts from decoded outputs.
    Keeps only the final assistant/model message content.
    """
    if not text:
        return text
    s = text
    # Prefer the last occurrence of a known assistant/model marker
    for marker in ["\nmodel\n", "model\n", "\nassistant\n", "assistant\n"]:
        idx = s.rfind(marker)
        if idx != -1:
            s = s[idx + len(marker) :]
            break
    # Drop possible leading role headers like 'user' leaked in output
    if s.startswith("user\n"):
        s = s.split("\n", 1)[-1]
    return s.strip()

def _combine_images_side_by_side(pil_a: Image.Image, pil_b: Image.Image) -> Image.Image:
    """Create a single image by placing A and B side-by-side with matched heights.
    Maintains aspect ratio for both, pads to white background if needed.
    """
    # Normalize to RGB
    a = pil_a.convert("RGB")
    b = pil_b.convert("RGB")
    # Match heights
    target_height = max(a.height, b.height)
    def resize_to_height(img: Image.Image, h: int) -> Image.Image:
        if img.height == h:
            return img
        w = int(round(img.width * (h / img.height)))
        return img.resize((w, h))
    a_resized = resize_to_height(a, target_height)
    b_resized = resize_to_height(b, target_height)
    total_width = a_resized.width + b_resized.width
    canvas = Image.new("RGB", (total_width, target_height), color=(255, 255, 255))
    canvas.paste(a_resized, (0, 0))
    canvas.paste(b_resized, (a_resized.width, 0))
    return canvas

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    client = request.client.host if request.client else "-"
    method = request.method
    path = request.url.path
    content_length = request.headers.get("content-length", "0")
    logging.info(
        f"PY REQ_START method={method} path={path} client={client} len={content_length}"
    )
    try:
        response = await call_next(request)
        status = response.status_code
        dur_ms = int((time.perf_counter() - start) * 1000)
        logging.info(
            f"PY REQ_END method={method} path={path} status={status} dur_ms={dur_ms}"
        )
        return response
    except Exception as exc:
        dur_ms = int((time.perf_counter() - start) * 1000)
        logging.exception(
            f"PY REQ_ERR method={method} path={path} dur_ms={dur_ms} error={exc}"
        )
        raise


class ImageItem(BaseModel):
    # One of these should be provided
    path: str | None = None
    url: str | None = None
    b64: str | None = None


class ImagesPayload(BaseModel):
    # Accept strings (backward compatible base64, paths, or URLs) or typed objects
    before: List[str | ImageItem] = []
    during: List[str | ImageItem] = []
    after: List[str | ImageItem] = []


class ScoreRequest(BaseModel):
    description: str
    summary: str | None = None
    images: ImagesPayload
    # Performance controls (optional)
    skip_descriptions: bool | None = None
    skip_pairwise: bool | None = None
    limit_per_group: int | None = None
    # Prompt controls (optional)
    # When true, the technician summary is ignored by the model prompts
    skip_summary: bool | None = None


def _b64_to_pil_image(b64_str: str) -> Image.Image:
    data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_image_from_value(val: Any) -> Image.Image:
    """Load a PIL image from a value that can be:
    - dict-like with keys: b64, url, path
    - string: http(s) URL, file path, or base64
    """
    try:
        if isinstance(val, dict):
            b64 = val.get("b64")
            url = val.get("url")
            path = val.get("path")
            if b64:
                return _b64_to_pil_image(b64)
            if url:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            if path:
                with open(path, "rb") as f:
                    return Image.open(io.BytesIO(f.read())).convert("RGB")
            raise ValueError("Unsupported image object; expected one of b64/url/path")

        if isinstance(val, Image.Image):
            return val

        if isinstance(val, str):
            s = val.strip()
            # URL?
            parsed = urlparse(s)
            if parsed.scheme in ("http", "https"):
                resp = requests.get(s, timeout=30)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            if parsed.scheme == "file":
                file_path = parsed.path
                with open(file_path, "rb") as f:
                    return Image.open(io.BytesIO(f.read())).convert("RGB")
            # Local path?
            if os.path.exists(s):
                with open(s, "rb") as f:
                    return Image.open(io.BytesIO(f.read())).convert("RGB")
            # Fallback: assume base64
            return _b64_to_pil_image(s)

        raise ValueError("Unsupported image value type")
    except Exception as exc:
        raise ValueError(f"Failed to load image: {exc}")


def _open_image_preserve_exif(data: bytes) -> tuple[Image.Image, Dict[str, Any]]:
    """Open image from bytes, capture EXIF before RGB conversion, return (rgb_img, exif_fields)."""
    img = Image.open(io.BytesIO(data))
    exif_fields = extract_exif_fields(img)
    rgb = img.convert("RGB")
    return rgb, exif_fields


def _load_image_and_exif_from_value(val: Any) -> tuple[Image.Image, Dict[str, Any]]:
    """Like _load_image_from_value, but also returns minimal EXIF fields.
    The returned image is RGB for model consumption.
    """
    try:
        if isinstance(val, dict):
            b64 = val.get("b64")
            url = val.get("url")
            path = val.get("path")
            if b64:
                data = base64.b64decode(b64)
                return _open_image_preserve_exif(data)
            if url:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                return _open_image_preserve_exif(resp.content)
            if path:
                with open(path, "rb") as f:
                    return _open_image_preserve_exif(f.read())
            raise ValueError("Unsupported image object; expected one of b64/url/path")

        if isinstance(val, Image.Image):
            # No EXIF available if already converted; best-effort extract from this instance
            return val.convert("RGB"), extract_exif_fields(val)

        if isinstance(val, str):
            s = val.strip()
            parsed = urlparse(s)
            if parsed.scheme in ("http", "https"):
                resp = requests.get(s, timeout=30)
                resp.raise_for_status()
                return _open_image_preserve_exif(resp.content)
            if parsed.scheme == "file":
                file_path = parsed.path
                with open(file_path, "rb") as f:
                    return _open_image_preserve_exif(f.read())
            if os.path.exists(s):
                with open(s, "rb") as f:
                    return _open_image_preserve_exif(f.read())
            # Fallback: assume base64
            data = base64.b64decode(s)
            return _open_image_preserve_exif(data)

        raise ValueError("Unsupported image value type")
    except Exception as exc:
        raise ValueError(f"Failed to load image: {exc}")


class GemmaScorer:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        auth = HF_TOKEN if HF_TOKEN else None
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, token=auth)
        # We still move input tensors to self.device for performance and to avoid warnings.
        self._use_device_map = self.device == "cuda"
        self.model = Gemma3nForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto" if self._use_device_map else None,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            token=auth,
        )

    @torch.no_grad()
    def describe_image(self, pil_img: Image.Image) -> tuple[str, str]:
        instruction = (
            "You are a vision system. Describe concisely what is visibly present in the photo "
            "(objects, scene, any work context). One short sentence, <= 50 words."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": "Provide a single sentence description."},
                ],
            },
        ]
        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        if "pixel_values" not in inputs:
            img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            for k, v in img_inputs.items():
                inputs[k] = v
        generated = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)
        raw = self.processor.decode(generated[0], skip_special_tokens=True).strip()
        out = _clean_chat_output(raw)
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"PY GEN_DESCRIBE dur_ms={dur_ms}")
        return out, raw

    @torch.no_grad()
    def score_image_text(
        self,
        pil_img: Image.Image,
        description: str,
        summary: str | None = None,
        stage: str | None = None,
    ) -> tuple[float, str]:
        stage_label = (stage or "unspecified").lower()
        # Comprehensive, stage-aware evaluation rubric
        refer_summary = bool(summary and str(summary).strip())
        prompt_instruction = (
            "You are a strict field-service photo compliance and relevancy evaluator for maintenance jobs. "
        )
        if refer_summary:
            prompt_instruction += (
                "Score how well the photo documents the specified stage of the job in relation to the provided "
                "job description and scope, and the technician's completion summary.\n\n"
            )
        else:
            prompt_instruction += (
                "Score how well the photo documents the specified stage of the job in relation to the provided "
                "job description and scope only.\n\n"
            )
        prompt_instruction += (
            "General guidance:\n"
            "- Base your decision ONLY on visual evidence in the photo.\n"
            "- Use the job description/scope"
            + (" and technician summary" if refer_summary else "")
            + " only as reference criteria; do not hallucinate missing details.\n"
            "- If the image is unrelated, too zoomed-out/in to verify, shows people/selfies without task context, or is a blank/screenshot, score low.\n\n"
            "Stage-specific expectations:\n"
            "- BEFORE: Expect clear capture of the target asset/location/problem area before work begins; identifiers, condition, surroundings. Penalize if it shows finished work.\n"
            "- DURING: Expect active work evidence (tools, parts, process steps, safety/PPE, progressive changes). Penalize if it looks fully finished or unrelated.\n"
            "- AFTER: Expect clear outcomes of completed work (repairs/installations complete, cleanliness, labeling, readings as applicable). Penalize if mid-work or broken/incomplete.\n\n"
            "Scoring rubric (integer 0–100):\n"
            "0=irrelevant, 25=weak/tangential, 50=partially relevant, 75=highly relevant with minor misses, 100=perfectly relevant.\n\n"
            "Return ONLY the integer score. No words or units."
        )

        summary_block = f"Technician Job Summary:\n{summary}\n\n" if refer_summary else ""
        tail_compare_clause = (
            "with respect to the job description/scope and the technician summary.\n"
            if refer_summary
            else "with respect to the job description and scope.\n"
        )
        user_text = (
            f"Stage: {stage_label}\n"
            f"Job Description and Scope:\n{description}\n\n"
            f"{summary_block}"
            f"Score this image for how well it documents the specified stage {tail_compare_clause}"
            "Answer with only a number."
        )

        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt_instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" not in inputs:
            img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            for k, v in img_inputs.items():
                inputs[k] = v
        generated = self.model.generate(**inputs, max_new_tokens=6, do_sample=False)
        out = self.processor.decode(generated[0], skip_special_tokens=True)
        import re
        nums = re.findall(r"(\d{1,3})", out)
        val = float(nums[-1]) if nums else 0.0
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"PY GEN_SCORE dur_ms={dur_ms}")
        return max(0.0, min(100.0, val)), out

    @torch.no_grad()
    def analyze_image_text(
        self,
        pil_img: Image.Image,
        description: str,
        summary: str | None = None,
        stage: str | None = None,
    ) -> tuple[str, str, str]:
        """Explain whether the image is relevant to the job description and summary for the stage.
        Returns (cleaned_explanation, verdict, raw_output).
        """
        stage_label = (stage or "unspecified").lower()
        refer_summary = bool(summary and str(summary).strip())
        if refer_summary:
            instruction = (
                "You are assessing how well a photo aligns with a maintenance job's description/scope and the technician's summary for a specific stage. "
                "Write 1–2 concise sentences citing visible evidence. Then add a verdict like 'Relevant', 'Partially', or 'Irrelevant'. "
                "Respond strictly in the format: <short explanation>\nVerdict: <Relevant|Partially|Irrelevant>."
            )
        else:
            instruction = (
                "You are assessing how well a photo aligns with a maintenance job's description/scope for a specific stage. "
                "Write 1–2 concise sentences citing visible evidence. Then add a verdict like 'Relevant', 'Partially', or 'Irrelevant'. "
                "Respond strictly in the format: <short explanation>\nVerdict: <Relevant|Partially|Irrelevant>."
            )
        summary_block = f"Technician Job Summary:\n{summary}\n\n" if refer_summary else ""
        tail_compare_clause = (
            "relative to the job description and summary."
            if refer_summary
            else "relative to the job description/scope."
        )
        user_text = (
            f"Stage: {stage_label}\n"
            f"Job Description and Scope:\n{description}\n\n"
            f"{summary_block}"
            f"Explain whether this image documents the specified stage {tail_compare_clause}"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" not in inputs:
            img_inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
            for k, v in img_inputs.items():
                inputs[k] = v
        generated = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        raw = self.processor.decode(generated[0], skip_special_tokens=True)
        cleaned = _clean_chat_output(raw)
        import re
        m = re.search(r"Verdict:\s*(Relevant|Partially|Irrelevant)", cleaned, flags=re.IGNORECASE)
        verdict = m.group(1).capitalize() if m else ""
        # Strip trailing verdict line from explanation
        explanation = re.sub(r"\n?Verdict:.*$", "", cleaned, flags=re.IGNORECASE).strip()
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"PY GEN_ANALYZE dur_ms={dur_ms}")
        return explanation, verdict, raw

    @torch.no_grad()
    def score_image_image(self, pil_a: Image.Image, pil_b: Image.Image) -> float:
        instruction = (
            "Rate the visual similarity between the two images from 0 to 100. "
            "Return only the integer number."
        )
        # Combine images into one to avoid image-token mismatch issues
        combined = _combine_images_side_by_side(pil_a, pil_b)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": combined},
                    {"type": "text", "text": "Answer with only a number."},
                ],
            },
        ]
        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" not in inputs:
            img_inputs = self.processor(images=combined, return_tensors="pt").to(self.device)
            for k, v in img_inputs.items():
                inputs[k] = v
        generated = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)
        out = self.processor.decode(generated[0], skip_special_tokens=True)
        import re
        nums = re.findall(r"(\d{1,3})", out)
        val = float(nums[-1]) if nums else 0.0
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"PY GEN_IMGIMG dur_ms={dur_ms}")
        return max(0.0, min(100.0, val))

    @torch.no_grad()
    def score_and_explain_image_image(self, pil_a: Image.Image, pil_b: Image.Image) -> tuple[float, str, str]:
        instruction = (
            "Rate the visual similarity between the two images from 0 to 100, then provide a brief reason. "
            "Return strictly in the format: <number>|<short reason>."
        )
        combined = _combine_images_side_by_side(pil_a, pil_b)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instruction}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": combined},
                    {"type": "text", "text": "Respond like: 78|Both show the same meter from slightly different angles."},
                ],
            },
        ]
        t0 = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" not in inputs:
            img_inputs = self.processor(images=combined, return_tensors="pt").to(self.device)
            for k, v in img_inputs.items():
                inputs[k] = v
        generated = self.model.generate(**inputs, max_new_tokens=24, do_sample=False)
        out = self.processor.decode(generated[0], skip_special_tokens=True)
        import re
        m = re.match(r"\s*(\d{1,3})\s*\|\s*(.*)$", out)
        if m:
            score_val = float(m.group(1))
            reason = _clean_chat_output(m.group(2).strip())
        else:
            # Fallback: extract the last number; treat rest as reason
            nums = re.findall(r"(\d{1,3})", out)
            score_val = float(nums[-1]) if nums else 0.0
            reason = _clean_chat_output(out.strip())
        score_val = max(0.0, min(100.0, score_val))
        dur_ms = int((time.perf_counter() - t0) * 1000)
        logging.info(f"PY GEN_IMGIMG_EXPL dur_ms={dur_ms}")
        return score_val, reason, out


if HF_TOKEN:
    try:
        hf_login(token=HF_TOKEN)
    except Exception:
        pass

# Optional: pre-download any requested models to speed up cold start

_scorer = GemmaScorer()
_active_vlm_label = MODEL_ID


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "vlm": _active_vlm_label,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/score")
def score(payload: ScoreRequest) -> Dict[str, Any]:
    # Request-level summary logs (avoid logging sensitive content)
    try:
        before_cnt = len(payload.images.before or [])
        during_cnt = len(payload.images.during or [])
        after_cnt = len(payload.images.after or [])
        desc_len = len(payload.description or "")
        summ_len = len(payload.summary or "")
        logging.info(
            f"PY SCORE_START before={before_cnt} during={during_cnt} after={after_cnt} desc_len={desc_len} summ_len={summ_len}"
        )
    except Exception:
        logging.info("PY SCORE_START (counts unavailable)")

    # Apply limits/flags
    limit = payload.limit_per_group if (payload.limit_per_group and payload.limit_per_group > 0) else None
    skip_desc = bool(payload.skip_descriptions)
    skip_pairs = bool(payload.skip_pairwise)
    # Determine whether to use the technician summary in prompts
    use_summary_flag = not bool(payload.skip_summary)
    effective_summary = payload.summary if use_summary_flag else None

    def score_group(items_list: List[Any], stage_label: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        sliced = items_list[:limit] if limit else items_list
        if limit is not None and len(items_list) > limit:
            logging.info(f"PY GROUP_LIMIT stage={stage_label} original={len(items_list)} used={len(sliced)}")
        for idx, item in enumerate(sliced):
            try:
                logging.info(
                    f"PY IMG_START stage={stage_label} idx={idx}"
                )
                img, exif_info = _load_image_and_exif_from_value(item)
                text_relevancy, raw_text_relevancy = _scorer.score_image_text(
                    img,
                    description=payload.description,
                    summary=effective_summary,
                    stage=stage_label,
                )
                vision_summary = None
                if not skip_desc:
                    vision_summary, raw_vision_summary = _scorer.describe_image(img)
                else:
                    logging.info(f"PY IMG_DESC_SKIPPED stage={stage_label} idx={idx}")
                analysis_expl, analysis_verdict, analysis_raw = _scorer.analyze_image_text(
                    img,
                    description=payload.description,
                    summary=effective_summary,
                    stage=stage_label,
                )
                record: Dict[str, Any] = {
                    "text_relevancy": text_relevancy,
                    "analysis": {
                        "explanation": analysis_expl,
                        "verdict": analysis_verdict,
                    },
                }
                if exif_info:
                    record["exif"] = exif_info
                if vision_summary is not None:
                    record["vision_summary"] = vision_summary
                    record["_raw"] = {
                        "text_relevancy": raw_text_relevancy,
                        "vision_summary": raw_vision_summary,
                        "analysis": analysis_raw,
                    }
                results.append(record)
                logging.info(
                    f"PY IMG_END stage={stage_label} idx={idx} text_relevancy={text_relevancy:.1f}"
                )
            except Exception as e:
                logging.warning(
                    f"PY IMG_ERR stage={stage_label} idx={idx} error={e}"
                )
                results.append({"error": str(e)})
        return results

    groups = {
        "before": score_group(payload.images.before, "before"),
        "during": score_group(payload.images.during, "during"),
        "after": score_group(payload.images.after, "after"),
    }

    # Image-image comparisons: compare before vs during/after for topical relevance
    def compare_pairs_detailed(a_list: List[Any], b_list: List[Any]) -> List[List[Dict[str, Any]]]:
        matrix: List[List[Dict[str, Any]]] = []
        if skip_pairs:
            logging.info("PY PAIRWISE_SKIPPED")
            return matrix
        pil_a_list = []
        a_iter = a_list[:limit] if limit else a_list
        b_iter = b_list[:limit] if limit else b_list
        if limit is not None:
            logging.info(
                f"PY PAIRWISE_LIMIT a_orig={len(a_list)} b_orig={len(b_list)} a_used={len(a_iter)} b_used={len(b_iter)}"
            )
        for a in a_iter:
            try:
                pil_img, _ = _load_image_and_exif_from_value(a)
                pil_a_list.append(pil_img)
            except Exception:
                pil_a_list.append(None)
        pil_b_list = []
        for b in b_iter:
            try:
                pil_img, _ = _load_image_and_exif_from_value(b)
                pil_b_list.append(pil_img)
            except Exception:
                pil_b_list.append(None)
        for pa in pil_a_list:
            row: List[Dict[str, Any]] = []
            for pb in pil_b_list:
                if pa is None or pb is None:
                    row.append({"score": float("nan"), "explanation": "invalid image"})
                else:
                    score_val, explanation, raw = _scorer.score_and_explain_image_image(pa, pb)
                    row.append({"score": score_val, "explanation": explanation, "_raw": raw})
            matrix.append(row)
        return matrix

    before_vs_during_details = compare_pairs_detailed(payload.images.before, payload.images.during)
    before_vs_after_details = compare_pairs_detailed(payload.images.before, payload.images.after)

    # Derive numeric matrices for backward compatibility
    def just_scores(detail_matrix: List[List[Dict[str, Any]]]) -> List[List[float]]:
        return [[cell.get("score", float("nan")) for cell in row] for row in detail_matrix]

    before_during = just_scores(before_vs_during_details)
    before_after = just_scores(before_vs_after_details)

    # Aggregate relevancy summaries
    def summarize_scores(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        vals = [x.get("text_relevancy") for x in items if isinstance(x.get("text_relevancy"), (int, float))]
        if not vals:
            return {"count": 0, "mean": float("nan")}
        return {"count": len(vals), "mean": float(sum(vals) / len(vals))}

    relevancy_summary = {
        "before": summarize_scores(groups["before"]),
        "during": summarize_scores(groups["during"]),
        "after": summarize_scores(groups["after"]),
    }
    # overall mean across all groups
    all_vals = []
    for g in ("before", "during", "after"):
        all_vals.extend([x.get("text_relevancy") for x in groups[g] if isinstance(x.get("text_relevancy"), (int, float))])
    overall_mean = float(sum(all_vals) / len(all_vals)) if all_vals else float("nan")

    # Build model_analysis object mirroring groups and pairwise with raw outputs
    model_analysis = {
        "groups": groups,
        "pairwise": {
            "before_vs_during": before_vs_during_details,
            "before_vs_after": before_vs_after_details,
        },
    }

    response: Dict[str, Any] = {
        "model": _active_vlm_label,
        "job": {
            "description": payload.description,
            "summary": payload.summary,
        },
        "groups": groups,
        "relevancy_summary": {
            "per_group": relevancy_summary,
            "overall_mean": overall_mean,
        },
        "image_pair_similarity": {
            "before_vs_during": before_during,
            "before_vs_after": before_after,
        },
        "image_pair_similarity_detailed": {
            "before_vs_during": before_vs_during_details,
            "before_vs_after": before_vs_after_details,
        },
        "model_analysis": model_analysis,
    }
    try:
        logging.info("PY SCORE_RESPONSE %s", json.dumps(response, ensure_ascii=False))
    except Exception as e:
        logging.warning("PY SCORE_RESPONSE_ERR %s", e)
    try:
        logging.info(
            f"PY SCORE_END before={before_cnt} during={during_cnt} after={after_cnt}"
        )
    except Exception:
        pass
    return response


