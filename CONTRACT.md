# Contract -- Averyn Face Service <-> NestJS Gateway

Version del contrato: 0.1.0
Servicio: Averyn Face Service (FastAPI, puerto 8001 interno)
Consumidor: NestJS API Gateway (proyecto principal Averyn)

---

## Base URL

```
http://face-service:8001
```

Acceso exclusivo via red interna Docker. No expuesto a internet.

---

## Endpoints

### `GET /health`

Healthcheck basico para orquestador de contenedores.

**Response 200:**
```json
{ "status": "ok" }
```

---

### `GET /api/v1/face/health`

Healthcheck detallado: verifica que los modelos de IA estan cargados.

**Response 200 (healthy):**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "concurrent_max": 2
}
```

**Response 200 (degraded):**
```json
{
  "status": "degraded",
  "models_loaded": false,
  "concurrent_max": 2
}
```

---

### `POST /api/v1/face/enroll`

Genera el embedding facial de 512 dimensiones a partir de una foto.
El gateway debe almacenar el embedding devuelto (ej. en Supabase/pgvector).

**Request:** `multipart/form-data`

| Campo | Tipo | Requerido | Descripcion |
|---|---|---|---|
| `imagen` | file | Si | Foto del rostro (JPEG/PNG) |

**Response 200:**
```json
{
  "success": true,
  "data": {
    "faceDetected": true,
    "faceCount": 1,
    "qualityOk": true,
    "qualityIssues": [],
    "isLive": true,
    "livenessConfidence": 0.9912,
    "embedding": [0.0012, -0.0345, ...]
  },
  "meta": {
    "model": "ArcFace",
    "detector": "yunet",
    "processingTimeMs": 412.3
  },
  "error": null
}
```

**Response 422 (validation error):**
El campo `imagen` no fue enviado o es invalido.

**Response 503 (backpressure):**
```json
{
  "success": false,
  "data": null,
  "meta": { "model": "ArcFace", "detector": "yunet", "processingTimeMs": 0 },
  "error": "Capacidad maxima alcanzada. Intente de nuevo."
}
```

**NOTA:** El `embedding` solo se devuelve cuando `isLive=true` y `qualityOk=true`.
Si la foto no pasa calidad o liveness, `embedding` es `null`.

---

### `POST /api/v1/face/verify`

Verificacion facial 1:1. Compara la foto contra un embedding registrado.

**Request:** `multipart/form-data`

| Campo | Tipo | Requerido | Descripcion |
|---|---|---|---|
| `imagen` | file | Si | Foto del rostro (JPEG/PNG) |
| `embedding_registrado` | text | No | JSON array de 512 floats |

**Response 200 (con embedding_registrado):**
```json
{
  "success": true,
  "data": {
    "faceDetected": true,
    "faceCount": 1,
    "qualityOk": true,
    "qualityIssues": [],
    "isLive": true,
    "livenessConfidence": 0.9912,
    "similarity": 0.8321,
    "verified": true
  },
  "meta": {
    "model": "ArcFace",
    "detector": "yunet",
    "processingTimeMs": 412.3
  },
  "error": null
}
```

**Response 200 (sin embedding_registrado -- solo calidad+liveness):**
Igual pero con `similarity: null` y `verified: null`.

**Response 400:**
```json
{
  "success": false,
  "data": null,
  "meta": { "model": "ArcFace", "detector": "yunet", "processingTimeMs": 0 },
  "error": "embedding_registrado no es un JSON valido."
}
```

**Response 503 (backpressure):**
Igual que enroll.

---

## Codigo de respuesta HTTP

| HTTP | Significado | Cuándo |
|---|---|---|
| 200 | Pipeline ejecutado | Siempre que el pipeline corra (calidad ok o no, liveness ok o no). `success=true`. |
| 400 | Parametro invalido | `embedding_registrado` no es JSON valido. `success=false`. |
| 422 | Validation error | Campo `imagen` faltante o tipo incorrecto (FastAPI automatico). |
| 503 | Capacidad agotada | Semaforo de concurrencia lleno. `success=false`. Reintentar con backoff. |
| 500 | Error tecnico | Error inesperado del servicio (no deberia ocurrir). `success=false`. |

**Importante para el diseno del adapter NestJS:**

- HTTP 200 + `success=true` + `qualityOk=false` = rechazo de negocio (no reintentar,
  devolver error de calidad al usuario).
- HTTP 200 + `success=true` + `isLive=false` = rechazo de liveness (no reintentar,
  pedir nueva captura).
- HTTP 200 + `success=true` + `verified=false` = verificacion fallida (decision de
  negocio, no error tecnico).
- HTTP 400 = error del caller (no reintentar, revisar el payload).
- HTTP 503 = sobrecarga (SI reintentar con backoff exponencial, max 3 reintentos).
- HTTP 500 = error tecnico (reintentar una vez, si persiste escalar).

---

## SLA medido (Fase 0/1 load test, 2026-09-16)

Numeros reales obtenidos con Locust en maquina local CPU-only, 3 workers,
`FACE_MAX_CONCURRENT_REQUESTS=1`:

| Metrica | Valor medido | Nota |
|---|---|---|
| P50 latencia (1 VU) | 350 ms | Procesado real de un request |
| P95 latencia (1 VU) | 760 ms | Procesado real de un request |
| P99 latencia (1 VU) | 970 ms | Procesado real de un request |
| Max throughput util (200s) | ~3 req/s | 3 workers x 1 concurrent. Limitado por el modelo, no por HTTP. |
| Throughput total (100 VU) | 234 req/s | La mayoria son 503 fail-fast (backpressure intencional) |
| Errores / timeouts | 0 / 0 | Tras el fix de Fase 1 (lock de inferencia) |
| RAM por worker | ~1.0 GB | Modelos completos en memoria |
| Etapa dominante | embedding | 82% del tiempo de pipeline (577 ms) |

Concurrencia real por replica: 1 (los modelos DeepFace no son thread-safe;
el lock en pipeline.py serializa la inferencia). Escalar via `--workers N`.

---

## Timeouts y circuit breaker (NestJS)

| Configuracion | Valor recomendado | Justificacion |
|---|---|---|
| HTTP timeout | 10 s | P99 estimado ~2 s, margen para cold start |
| Circuit breaker threshold | 5 fallos en 10 s | Evitar cascada de reintentos |
| Circuit breaker timeout | 30 s | Tiempo para que el servicio se recupere |
| Max retries (503) | 3 | Backoff exponencial: 1s, 2s, 4s |
| Max retries (500) | 1 | Si persiste, escalar |

---

## Sizing de replicas

| Escenario | Replicas face-service | RAM por replica |
|---|---|---|
| Baja carga (< 100 verificaciones/dia) | 1 | ~1.0-1.5 GB (medido: 1.0 GB) |
| Media carga (100-1000/dia) | 1-2 | ~1.0-1.5 GB c/u |
| Alta carga (> 1000/dia) | 2-3 + cola async | ~1.0-1.5 GB c/u |

**NOTA:** Cada replica carga los modelos completos en RAM (~1.0 GB medido, no
~2-3 GB como se estimaba). El concurrency limit (por defecto 1) controla cuantos
requests procesa simultaneamente por replica; los modelos no son thread-safe,
asi que escalar significa mas workers, no mas concurrencia por worker.

---

## Variables de entorno

| Variable | Default | Descripcion |
|---|---|---|
| `FACE_MODEL_NAME` | `ArcFace` | Modelo de embeddings |
| `FACE_DETECTOR_BACKEND` | `yunet` | Detector de rostros |
| `FACE_SIMILARITY_THRESHOLD` | `0.68` | Umbral de verificacion (sin calibrar) |
| `FACE_MAX_CONCURRENT_REQUESTS` | `1` | Concurrency max por replica (los modelos no son thread-safe; escalar con workers) |
| `FACE_BRILLO_MIN` | `60` | Brillo minimo de imagen |
| `FACE_BRILLO_MAX` | `200` | Brillo maximo de imagen |
| `FACE_NITIDEZ_MIN` | `20.0` | Nitidez minima (Laplaciano) |
| `FACE_ANCHO_MIN` | `200` | Ancho minimo de imagen (px) |
| `FACE_ALTO_MIN` | `200` | Alto minimo de imagen (px) |
| `FACE_ROSTRO_AREA_MIN_RATIO` | `0.05` | Area minima del rostro (ratio) |
| `FACE_ROSTRO_ANCHO_MIN_PX` | `50` | Ancho minimo del rostro (px) |
| `FACE_HOST` | `0.0.0.0` | Host de escucha |
| `FACE_PORT` | `8001` | Puerto de escucha |
