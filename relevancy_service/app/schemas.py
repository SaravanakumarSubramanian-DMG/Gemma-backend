from typing import List, Optional, Any
from pydantic import BaseModel


class HealthResponse(BaseModel):
	status: str


class InferenceMeta(BaseModel):
	inference_ms: float
	text: str
	image_url: Optional[str] = None
	model: str
	vision_summary: Optional[str] = None
	relevancy_reason: Optional[str] = None


class RelevancyRequest(BaseModel):
	text: str
	image_base64: Optional[str] = None
	image_url: Optional[str] = None


class RelevancyResponse(BaseModel):
	score: float
	meta: InferenceMeta


class BatchItem(BaseModel):
	text: str
	image_base64: Optional[str] = None
	image_url: Optional[str] = None


class BatchRelevancyRequest(BaseModel):
	items: List[BatchItem]


class BatchRelevancyItemResponse(BaseModel):
	score: float
	meta: InferenceMeta


class BatchRelevancyResponse(BaseModel):
	results: List[BatchRelevancyItemResponse]


class TextEncodeRequest(BaseModel):
	texts: List[str]


class ImageEncodeRequest(BaseModel):
	images_base64: Optional[List[str]] = None
	images_url: Optional[List[str]] = None


class EmbeddingsResponse(BaseModel):
	embeddings: List[List[float]]
	model: str


class ImageItem(BaseModel):
	path: Optional[str] = None
	url: Optional[str] = None
	b64: Optional[str] = None


class ImagesPayload(BaseModel):
	before: List[Any] = []
	during: List[Any] = []
	after: List[Any] = []


class ScoreRequest(BaseModel):
	description: str
	summary: Optional[str] = None
	images: ImagesPayload
	skip_descriptions: Optional[bool] = None
	skip_pairwise: Optional[bool] = None
	skip_summary: Optional[bool] = None


class ScoreCell(BaseModel):
	score: float
	explanation: Optional[str] = None
	_raw: Optional[str] = None


class ScoreGroupItem(BaseModel):
	text_relevancy: Optional[float] = None
	vision_summary: Optional[str] = None
	exif: Optional[dict] = None
	analysis: Optional[dict] = None
	_raw: Optional[dict] = None
	error: Optional[str] = None


class ScoreResponse(BaseModel):
	model: str
	job: dict
	groups: dict
	relevancy_summary: dict
	image_pair_similarity: dict
	image_pair_similarity_detailed: dict
	model_analysis: dict

	class Config:
		protected_namespaces = ()
