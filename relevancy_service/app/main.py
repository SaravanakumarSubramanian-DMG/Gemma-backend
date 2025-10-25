from typing import List
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from .schemas import (
	BatchRelevancyRequest,
	BatchRelevancyResponse,
	BatchRelevancyItemResponse,
	EmbeddingsResponse,
	HealthResponse,
	ImageEncodeRequest,
	RelevancyRequest,
	RelevancyResponse,
	TextEncodeRequest,
	ScoreRequest,
	ScoreResponse,
)
from .service import RelevancyService
from .utils.image import load_image_from_base64, load_image_from_url, load_images_from_base64_list, load_images_from_url_list
from .config import settings

logger = logging.getLogger("relevancy_service.api")

app = FastAPI(default_response_class=ORJSONResponse, title="Relevancy Service", version="1.0.0")
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

service = RelevancyService()
logger.info("service_started model=%s device=%s hf_token=%s", service.model_name, settings.device, "set" if settings.hf_token else "unset")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
	return HealthResponse(status="ok")


@app.post("/relevancy", response_model=RelevancyResponse)
@app.post("/predict", response_model=RelevancyResponse)
def relevancy(req: RelevancyRequest) -> RelevancyResponse:
	if not req.text:
		raise HTTPException(status_code=400, detail="text is required")
	if not req.image_base64 and not req.image_url:
		raise HTTPException(status_code=400, detail="image_base64 or image_url is required")
	image = load_image_from_base64(req.image_base64) if req.image_base64 else load_image_from_url(req.image_url or "")
	res = service.score_single_with_meta(req.text, image, image_url=req.image_url)
	return RelevancyResponse(**res)


@app.post("/batch_relevancy", response_model=BatchRelevancyResponse)
def batch_relevancy(req: BatchRelevancyRequest) -> BatchRelevancyResponse:
	texts: List[str] = []
	images = []
	urls: List[str] = []
	for item in req.items:
		if not item.text:
			raise HTTPException(status_code=400, detail="text is required for all items")
		if not item.image_base64 and not item.image_url:
			raise HTTPException(status_code=400, detail="image_base64 or image_url is required for all items")
		texts.append(item.text)
		urls.append(item.image_url or "")
		images.append(load_image_from_base64(item.image_base64) if item.image_base64 else load_image_from_url(item.image_url or ""))
	scores = service.score_pairs(texts, images)
	results = []
	for s, t, u in zip(scores, texts, urls):
		meta = {
			"inference_ms": 0.0,
			"text": t,
			"image_url": u or None,
			"model": service.model_name,
		}
		results.append(BatchRelevancyItemResponse(score=float(s), meta=meta))
	return BatchRelevancyResponse(results=results)


@app.post("/encode/text", response_model=EmbeddingsResponse)
def encode_text(req: TextEncodeRequest) -> EmbeddingsResponse:
	if not req.texts:
		raise HTTPException(status_code=400, detail="texts cannot be empty")
	emb = service.encode_texts(req.texts)
	return EmbeddingsResponse(embeddings=emb, model=service.model_name)


@app.post("/encode/image", response_model=EmbeddingsResponse)
def encode_image(req: ImageEncodeRequest) -> EmbeddingsResponse:
	images = []
	if req.images_base64:
		images.extend(load_images_from_base64_list(req.images_base64))
	if req.images_url:
		images.extend(load_images_from_url_list(req.images_url))
	if not images:
		raise HTTPException(status_code=400, detail="images_base64 or images_url required")
	emb = service.encode_images(images)
	return EmbeddingsResponse(embeddings=emb, model=service.model_name)


@app.post("/relevancy/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
	skip_desc = bool(req.skip_descriptions)
	skip_pairs = bool(req.skip_pairwise)
	use_summary_flag = not bool(req.skip_summary)
	effective_summary = req.summary if use_summary_flag else None

	groups = {
		"before": service.group_score_and_analysis(req.images.before or [], req.description, effective_summary, "before", skip_desc),
		"during": service.group_score_and_analysis(req.images.during or [], req.description, effective_summary, "during", skip_desc),
		"after": service.group_score_and_analysis(req.images.after or [], req.description, effective_summary, "after", skip_desc),
	}

	def summarize(values: List[float]) -> dict:
		if not values:
			return {"count": 0, "mean": float("nan")}
		return {"count": len(values), "mean": float(sum(values) / len(values))}

	def extract_scores(items: List[dict]) -> List[float]:
		return [float(x.get("text_relevancy")) for x in items if isinstance(x.get("text_relevancy"), (int, float))]

	relevancy_summary = {
		"before": summarize(extract_scores(groups["before"])),
		"during": summarize(extract_scores(groups["during"])),
		"after": summarize(extract_scores(groups["after"])),
	}
	all_vals = extract_scores(groups["before"]) + extract_scores(groups["during"]) + extract_scores(groups["after"])
	overall_mean = float(sum(all_vals) / len(all_vals)) if all_vals else float("nan")

	before_during, before_during_detailed = ([], [])
	before_after, before_after_detailed = ([], [])
	if not skip_pairs:
		before_during, before_during_detailed = service.pairwise_similarity_matrix(req.images.before or [], req.images.during or [])
		before_after, before_after_detailed = service.pairwise_similarity_matrix(req.images.before or [], req.images.after or [])

	return ScoreResponse(
		model=service.model_name,
		job={"description": req.description, "summary": req.summary},
		groups=groups,
		relevancy_summary={"per_group": relevancy_summary, "overall_mean": overall_mean},
		image_pair_similarity={"before_vs_during": before_during, "before_vs_after": before_after},
		image_pair_similarity_detailed={"before_vs_during": before_during_detailed, "before_vs_after": before_after_detailed},
		model_analysis={"groups": groups},
	)
