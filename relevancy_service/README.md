# Relevancy Service (Image ⟂ Text)

A production-ready FastAPI microservice for photo–job relevancy inference. It supports dual-encoder inference (SigLIP/CLIP), optional Gemma multimodal scoring, micro-batching, LRU caching, cosine similarity, and export/quantization tooling (ONNX → TFLite FP16, TensorRT).

## Architecture (Text Description)

- Ingress: React Native app sends `job_description` + `image` to `/relevancy`.
- API Layer: FastAPI validates input (Pydantic), authenticates (optional), routes requests.
- Embedding Layer (Dual-Encoder):
  - Text Encoder: SigLIP/CLIP text tower → text embedding
  - Vision Encoder: SigLIP/CLIP vision tower → image embedding
- Caching:
  - Text Embeddings: LRU cache keyed by SHA256 of normalized text
  - Image Embeddings: Optional LRU cache keyed by SHA256 of image bytes
- Similarity:
  - Cosine similarity → relevancy score in [0, 1]
- Micro-batching:
  - Queue aggregates requests within a short window and encodes as a batch
- Optional Cross-Modal Scoring (Gemma):
  - Prompt-based VLM scoring for tie-breaks or audits
- Serving:
  - Uvicorn or Gunicorn+Uvicorn workers
  - Optional ONNXRuntime/TensorRT engines

```
[Client] -> [FastAPI /relevancy] -> [Batch Queue]
                            |-> [Text Encoder] -> [Text Emb]
                            |-> [Vision Encoder] -> [Image Emb]
                                 [LRU Cache]                
                         [Cosine Similarity] -> score [0..1]
                         [Optional Gemma Cross-Modal Scorer]
```

## Endpoints

- `GET /health` – Service readiness
- `POST /relevancy` – Single image + text → score (accepts `image_base64` or `image_url`)
- `POST /predict` – Alias of `/relevancy`
- `POST /batch_relevancy` – Batch of (text, image) pairs → scores
- `POST /encode/text` – Text → embedding
- `POST /encode/image` – Image → embedding
- `POST /relevancy/score` – Group scoring and analysis for `before`/`during`/`after` images, with optional pairwise similarity

## Quickstart (GPU)

```bash
# 1) Create venv
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run server
uvicorn app.main:app --host 0.0.0.0 --port 8080 --workers 1
```

ENV options (defaults reasonable):
- `MODEL_NAME` (e.g., google/siglip-so400m-patch14-384)
- `DEVICE` (cuda|cpu; defaults to cuda if available)
- `IMAGE_SIZE` (e.g., 384)
- `BATCH_SIZE` (e.g., 16)
- `BATCH_MAX_DELAY_MS` (e.g., 8)
- `EMBED_CACHE_MAX_ITEMS` (e.g., 5000)
- `LOG_LEVEL` (debug|info|warning|error)
- `HF_TOKEN` (optional; Hugging Face token for gated models)
- `GEMMA_MODEL_NAME` (e.g., google/gemma-3-2b-it)
- `USE_GEMMA_EXPLANATIONS` (true|false; adds brief summaries/reasons)
- `PRIMARY_SCORER` (dual|gemma; choose dual-encoder vs VLM primary)

## Docker (Remote Deploy)

```bash
# Build (with NVIDIA GPU support)
docker build -t relevancy-service:latest .

# Run (GPU)
docker run --gpus all -p 8080:8080 \
  -e DEVICE=cuda \
  -e MODEL_NAME=google/siglip-so400m-patch14-384 \
  -e PRIMARY_SCORER=dual \
  -e USE_GEMMA_EXPLANATIONS=true \
  -e GEMMA_MODEL_NAME=google/gemma-3-2b-it \
  -e HF_TOKEN=$HF_TOKEN \
  relevancy-service:latest
```

## Export & Quantization

- ONNX Export: `python scripts/export_onnx.py`
- TFLite FP16: `python scripts/export_tflite.py` (via onnx2tf)
- TensorRT Engine: `python scripts/export_tensorrt.py`

## Benchmark

```bash
python scripts/benchmark.py --images_dir ./samples --text_file ./samples/job.txt
```

## Example Client Call

```bash
python scripts/example_client.py \
  --endpoint http://localhost:8080/relevancy \
  --image ./samples/site.jpg \
  --text "Trim hedges and clear debris around perimeter."
```

### Example: `/relevancy/score`

Request (any image item may be a dict with `b64`, `url`, or `path`, or a raw string of base64/URL/path):

```json
{
  "description": "Clean leaves from gutters and remove fallen branches.",
  "summary": "Gutters clear, branches removed",
  "images": {
    "before": [{ "url": "https://example.com/before1.jpg" }],
    "during": [],
    "after": [{ "b64": "<base64>" }]
  },
  "skip_descriptions": false,
  "skip_pairwise": false,
  "skip_summary": false
}
```

Response (abridged):

```json
{
  "model": "google/siglip-so400m-patch14-384",
  "job": { "description": "...", "summary": "..." },
  "groups": { "before": [{ "text_relevancy": 73.2, "vision_summary": "..." }], "during": [], "after": [] },
  "relevancy_summary": { "per_group": {"before": {"count": 1, "mean": 73.2} }, "overall_mean": 71.4 },
  "image_pair_similarity": { "before_vs_during": [[...]], "before_vs_after": [[...]] },
  "image_pair_similarity_detailed": { "before_vs_during": [[{ "score": 73.2 }]], "before_vs_after": [[{ "score": 70.0 }]] },
  "model_analysis": { "groups": { "before": [{ "_raw": { } }] } }
}
```

## Recommended Models

- Default: `google/siglip-so400m-patch14-384` (strong accuracy, good speed)
- Faster: `google/siglip-base-patch16-224` or `openai/clip-vit-base-patch32`
- Optional VLM (audit/tie-break): Gemma 3N IT multimodal variant

## Performance Targets

- Dual-encoder + GPU + batching: <300 ms/image typical
- FP16 or INT8 (TensorRT) recommended for tight SLAs
- Cache precompute text embeddings for repeated jobs

## Notes on Accuracy

- Use domain prompts to focus text (e.g., include task type + checklist)
- Consider ensemble: dual-encoder primary, VLM as fallback when score ~0.5–0.6

## SOLID Design

- `EncoderInterface` (models/base.py) isolates implementations
- Utilities are single-responsibility (image, text, cache, similarity)
- Dependency injection via `ModelLoader` and `Config`

