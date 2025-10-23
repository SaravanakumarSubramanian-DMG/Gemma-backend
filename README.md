## Gemma-based Image Relevancy Service

Node.js API (Express) + Python FastAPI inference service. The Node API accepts job description/summary and images grouped as before/during/after, forwards them to the Python scorer, and returns relevancy scores.

### Architecture
- Node API: `src/server.js`
- Python service: `python_service/app.py` (CLIP ViT-B/32 for image-text and image-image similarity)

You can later swap CLIP for Gemma-3 multimodal when an official vision-capable checkpoint is broadly available in `transformers` with stable APIs.

### Setup
1. Requirements
   - Node.js 18+
   - Python 3.10+
   - On GPU instances (recommended), install CUDA drivers for PyTorch.

2. Install Node deps
```bash
npm install
```

3. Create Python venv and install
```bash
npm run py:install
```

4. Environment
Create `.env` in project root:
```
PORT=3000
PY_SERVICE_URL=http://127.0.0.1:8000
HF_TOKEN=YOUR_HF_TOKEN
```

5. Start services (two terminals) or combined
```bash
# Terminal 1 - Python
npm run py:start

# Terminal 2 - Node
npm run dev

# Or both with auto-reload for Node
npm run start:all
```

### API
POST `/api/relevancy`
Form-data (multipart):
- `description`: string (required)
- `summary`: string (optional)
- `before`: files[]
- `during`: files[]
- `after`: files[]

Response JSON:
```json
{
  "model": "clip-vit-b-32",
  "groups": {
    "before": [{"text_relevancy": 77.1, "filename": "before1.jpg"}],
    "during": [{"text_relevancy": 81.5, "filename": "during1.jpg"}],
    "after": [{"text_relevancy": 85.2, "filename": "after1.jpg"}]
  },
  "image_pair_similarity": {
    "before_vs_during": [[72.3, 68.4]],
    "before_vs_after": [[70.1, 74.8]]
  }
}
```

Scores are scaled roughly to [-100, 100] as cosine*100 via CLIP; higher is more relevant/similar.

### Deploy to AWS EC2 (outline)
1. Provision an EC2 (GPU if possible, e.g., g5.xlarge). Open ports 22, 3000, 8000 (or place behind a reverse proxy).
2. SSH, install Node 18+, Python 3.10+, Git, and build tools.
3. Clone repo, set `.env` with `HF_TOKEN`.
4. `npm install` and `npm run py:install` (ensure CUDA-enabled torch if GPU).
5. Start Python: `HF_TOKEN=... npm run py:start` and Node: `npm run start` (or use `pm2`/systemd).
6. Put Nginx in front if desired and enable HTTPS.

### Notes on Gemma-3
If you want to swap to Gemma-3 multimodal when publicly available with stable image+text inference:
- Replace `python_service/app.py` scorer with the appropriate `transformers` vision-text model and processor.
- Use the same request/response schema so Node remains unchanged.


