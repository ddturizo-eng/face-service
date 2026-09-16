# Load Test Report -- Averyn Face Service

Fase 0/1 baseline measurements, taken on a local Windows machine (CPU-only,
no GPU). Reference numbers for Fase 6 SLA (timeouts, replicas, VPS sizing).

Test date: 2026-09-16
Load tool: Locust 2.46.5 (headless)
Service version: after Fase 1 changes (async fix + inference lock + 503 backpressure),
run with `uvicorn app.main:app --workers 3`
Test images: `test_images/img1.jpg` (real captures), 4 files loaded.

Request mix (locustfile.py): verify sin embedding (10), verify con embedding (5),
enroll (3), health checks (3).

---

## 1. Summary table

| VUs | Total requests | Failures | Processed req/s* | P50 | P95 | P99 | Max latency |
|-----|---------------|----------|------------------|-----|-----|-----|-------------|
| 1   | 51            | 0        | 1.76             | 350 ms | 760 ms | 970 ms | 967 ms |
| 10  | 970           | 0        | 33.3             | 14 ms  | 640 ms | 1500 ms | 1479 ms |
| 50  | 5113          | 0        | 178.3            | 37 ms  | 150 ms | 1600 ms | 2029 ms |
| 100 | 6827          | 0        | 234.0            | 110 ms | 390 ms | 1800 ms | 3162 ms |

*Processed req/s = total of 200s + 503s returned by the server. At high VU the
majority are fast 503 rejections (intentional backpressure), see breakdown below.

## 2. Where the system degrades today

- **Useful throughput is capped by the model, not the HTTP layer.** With
  `--workers 3` and `FACE_MAX_CONCURRENT_REQUESTS=1`, at most 3 requests run
  the full pipeline in parallel. Sustained processed throughput of **200-status
  requests saturates around 3 req/s** (3 workers x ~1/350 ms).
- **Degradation is graceful and measurable.** At 100 VUs the P50 latency is
  110 ms because most requests are rejected with 503 in milliseconds. The
  processed (200) requests see P99 ~3.2 s under full load (CPU contention
  between the 3 model instances on the same cores).
- **No 500s, no timeouts** in any scenario after the Fase 1 fix. Before the fix
  (threadpool without inference lock), concurrent model access produced
  `cv2.error ... buf.shape() == m.shape()` 500s.
- Rejection point: **capacity = workers x FACE_MAX_CONCURRENT_REQUESTS**.
  At 3 workers x 1 = 3. Every request beyond that immediately receives 503.

## 3. Pipeline stage breakdown (Prometheus histogram)

Measured from `/metrics` (`face_pipeline_*_ms_sum / _count`) across 142 full
pipeline completions under load.

| Stage | Avg time | Share of pipeline |
|-------|----------|-------------------|
| resize              | 13.6 ms | 1.9%  |
| calidad_imagen      | 8.8 ms  | 1.2%  |
| deteccion_y_liveness| 102.5 ms| 14.6% |
| calidad_rostro      | 0.03 ms | <0.1% |
| embedding (ArcFace) | 577.0 ms| 82.3% |

**Conclusion:** any latency optimization must target the embedding stage
(ArcFace represent). Detection + liveness is a distant second. Calidad checks
are negligible.

## 4. Resource usage

Measured with 3 uvicorn workers on the local machine:

| Resource | Value |
|----------|-------|
| RAM per worker | ~1.0 GB (models fully loaded) |
| Total RAM (3 workers) | ~3.0 GB |
| CPU | CPU-only inference, high usage during pipeline runs |

## 5. Recommendation (feeds Fase 6)

- **Replicas:** for the POC, 1-2 workers is enough (each worker adds ~1 GB RAM
  and +1 concurrent pipeline). 3 workers showed no additional practical value on
  this machine (CPU-bound).
- **Latency budget:** P95 ~760 ms at 1 VU; budget the NestJS HTTP timeout at
  10 s (covers SLO with margin, including cold start of a replica).
- **Circuit breaker:** 503s are the expected overflow signal. Configure NestJS
  to treat 503 as "retry with backoff" and 500 as "real failure, alert".
- **VPS sizing for CPU-only:** each worker needs ~1 GREGGI GB RAM + OS overhead.
  a 4 GB VPS runs 1-2 workers; 8 GB runs 3-4. GPU would move the bottleneck
  (embedding stage) off the CPU entirely.

## 6. How to reproduce

```bash
# Terminal 1: start the service (3 workers)
uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 3

# Terminal 2: run each scenario
locust -f locustfile.py --host http://127.0.0.1:8001 --headless -u 1  -r 1  --run-time 30s --csv=loadtest/1vu
locust -f locustfile.py --host http://127.0.0.1:8001 --headless -u 10 -r 10 --run-time 30s --csv=loadtest/10vu
locust -f locustfile.py --host http://127.0.0.1:8001 --headless -u 50 -r 50 --run-time 30s --csv=loadtest/50vu
locust -f locustfile.py --host http://127.0.0.1:8001 --headless -u 100 -r 100 --run-time 30s --csv=loadtest/100vu
```

For Docker deployments, run `docker stats --format "table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}"`
in parallel to capture container metrics instead of the local process table.