# ABIS Web — Face Service

> 🔬 **Status: POC (Proof of Concept)** — This project is a proof of concept.
> The face verification pipeline works, but it is **not production-ready**: enrolment
> via API, embeddings persistence (Postgres/Supabase + pgvector) and biometric data
> protection are still pending. See [Production roadmap](#production-roadmap-pending).

A [FastAPI](https://fastapi.tiangolo.com/) face recognition microservice that runs a
**1:1 verification** pipeline: capture quality → face detection + anti-spoofing →
embedding → comparison against a previously registered embedding.

Built on [DeepFace](https://github.com/serengil/deepface) and designed to run on
**CPU** (no GPU required) inside Docker.

## ✨ Features

- **Image quality**: rejects dark, overexposed or blurry captures (configurable thresholds).
- **Face quality**: rejects multiple faces and faces that are too small.
- **Anti-spoofing (liveness)**: distinguishes real photos from screens/photos of screens (MiniFAS).
- **ArcFace embeddings** (512-dim) with cosine-similarity verification.
- **Optimized**: detection + liveness in a single pass; the embedding is computed without re-detecting.
- **Model warm-up** on startup and during the Docker build (runtime starts with no internet access).
- **CPU-only Docker image** ready for resource-constrained hardware.

## 🧱 Architecture

```
                     ┌──────────────────────────────┐
   Client/API -----─►│  FastAPI (port 8001)         │
    (HTTP)           │  app/main.py                 │
                     │  ├─ GET  /health             │
                     │  └─ POST /api/v1/face/verify │
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  services/pipeline.py         │  Pipeline orchestration
                     ├── services/quality.py         │  Image + face quality
                     ├── core/config.py              │  Models, detector, thresholds
                     └──────────────┬───────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  DeepFace                    │
                     │  ├─ YuNet   (detector)       │
                     │  ├─ MiniFAS (liveness)       │
                     │  └─ ArcFace (embedding)      │
                     └─────────────────────────────┘
```

### Models

| Model | Purpose |
|---|---|
| **YuNet** (OpenCV) | Face detection (chosen for speed) |
| **ArcFace** | 512-dimensional embeddings |
| **MiniFAS** | Liveness / anti-spoofing |

## 🚀 Getting Started

### Local (Python 3.11)

```bash
git clone https://github.com/<your-user>/face-service.git
cd face-service
python -m venv venv
venv\Scripts\activate        # Windows  |  source venv/bin/activate (Linux/macOS)
pip install -r requirements.txt
uvicorn app.main:app --port 8001
```

> The first run downloads the models (~1-2 GB) into `~/.deepface` or `$DEEPFACE_HOME`;
> the service warm-ups all models on startup (~10-15 s).

### Docker

```bash
docker build -t abis-face-service .
docker run -p 8001:8001 abis-face-service
```

> Models are downloaded and cached inside the image at build time, so the container
> starts without any internet dependency.

## 📡 API

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /api/v1/face/verify`
`multipart/form-data`:
| Field | Type | Required | Description |
|---|---|---|---|
| `imagen` | file | ✅ | Photo to analyze |
| `embedding_registrado` | text | ❌ | JSON array of floats (the registered person's embedding) |

**Response:**
```json
{
  "success": true,
  "data": {
    "faceDetected": true,
    "faceCount": 1,
    "qualityOk": true,
    "qualityIssues": [],
    "isLive": true,
    "livenessConfidence": 0.99,
    "similarity": 0.8321,
    "verified": true
  },
  "meta": { "model": "ArcFace", "detector": "yunet", "processingTimeMs": 412.3 },
  "error": null
}
```

**Early-rejection codes** (all return `verified: null`):

| `qualityIssues` | Reason |
|---|---|
| `resolucion_insuficiente (WxH)` | Image smaller than 200×200 px |
| `imagen_muy_oscura` / `imagen_sobreexpuesta` | Mean brightness outside [60, 200] |
| `imagen_borrosa` | Laplacian variance < 20 |
| `ningun_rostro_detectado` | No face found |
| `multiples_rostros_detectados (N)` | More than one face in the image |
| `rostro_muy_pequeno` | Face < 5% of image area or < 50 px wide |

### Example with curl

```bash
curl -X POST http://localhost:8001/api/v1/face/verify \
  -F "imagen=@path/to/photo.jpg" \
  -F "embedding_registrado=[0.0012,-0.0345,...]"   # 512 floats
```

## ⚙️ Configuration

All thresholds live in [`app/core/config.py`](app/core/config.py):

| Constant | Value | Description |
|---|---|---|
| `MODEL_NAME` | `ArcFace` | Embedding model |
| `DETECTOR_BACKEND` | `yunet` | Face detector |
| `BRILLO_MIN` / `BRILLO_MAX` | 60 / 200 | Image quality |
| `NITIDEZ_MIN` | 20.0 | Image quality (Laplacian) |
| `ANCHO_MIN` / `ALTO_MIN` | 200 / 200 | Minimum resolution (px) |
| `ROSTRO_AREA_MIN_RATIO` | 0.05 | Face quality |
| `ROSTRO_ANCHO_MIN_PX` | 50 | Face quality |
| `SIMILARITY_THRESHOLD` | 0.68 | Verification (cosine) — **not calibrated** |

> ⚠️ Thresholds are **starting values** and `SIMILARITY_THRESHOLD` is DeepFace's
> documented default; recalibrate with real data (FAR/FRR) before production.

## 🧪 Test scripts

- [`test_comparacion_detectores.py`](test_comparacion_detectores.py) — compares
  RetinaFace vs. YuNet on pairs from the LFW dataset (requires downloading
  [LFW](https://vis-www.cs.umass.edu/lfw/) into `test_images/lfw_funneled`).

## 📁 Structure

```
app/
├── main.py                    # API: /health and /api/v1/face/verify
├── core/config.py             # Models, detector and thresholds
├── schemas/face.py            # Response schemas
└── services/
    ├── pipeline.py            # Pipeline orchestration
    └── quality.py             # Image and face quality
dockerfile                     # CPU-only image with preloaded models
requirements.txt               # Dependencies
warmup.py                      # Download/cache models at Docker build time
```

## 🔒 Privacy note

`test_images/` is **excluded** from version control (`.gitignore` / `.dockerignore`)
because it may contain real face photos, which are sensitive personal data.
**Never commit real photos or any other sensitive/anonymizeable data to this repo.**

## 🗺️ Production roadmap (pending)

- **Enrolment**: the endpoint generates the embedding in "enrolment mode" but does
  not return it in the response; it must be exposed to register new embeddings.
- **Database**: no persistence yet; planned to store embeddings in Postgres
  (Supabase) with `pgvector`.
- **Biometric data protection**: encryption at rest, RLS, consent and deletion
  (embeddings are sensitive data).
- **Security**: authentication, rate limiting, TLS at deployment.
- **Threshold calibration** with real data.
- **Concurrency**: the endpoints block the event loop (synchronous CPU work).
- **Automated tests** and CI.

## 📄 License

MIT — see [LICENSE](LICENSE).