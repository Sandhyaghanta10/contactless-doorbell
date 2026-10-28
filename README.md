# Contactless Doorbell - face recognition on a budget

A doorbell that watches a camera, recognizes registered visitors, and emails the homeowner,
with a benchmark that measures how recognition **accuracy, latency and lighting** trade off.

> Recreated in 2026. Originally built 2018–22.

## The original project
The original was a two-person B.Tech project report at SSN College of Engineering, built with **Boyapati Yagnavi Chowdari**.
It was a Raspberry Pi 3B+ doorbell with a webcam, a buzzer and a 16x2 I2C LCD:
- A registered visitor is greeted on the LCD and the owner gets an email with the visitor's name and photo.
- An unknown visitor sounds the buzzer and the owner gets an email with their photo.
- Faces were matched with the `face_recognition` library (dlib 128-d embeddings) over an OpenCV Haar-cascade detector; email went out via Mailgun.
- **Result reported:** 91.4% accuracy on a known-vs-unknown test (confusion matrix: 58 unknown correct, 10 unknown wrongly accepted,
  4 known wrongly rejected, 92 known correct = 150 / 164). The report does not describe the test set beyond this matrix.

## How this recreation differs
This repository is a fresh software-only rewrite, not the original code. It is **not** a reproduction of the original's results.

| | Original (2022 report) | This recreation (2026) |
|---|---|---|
| Hardware | Raspberry Pi 3B+, webcam, buzzer, LCD | None; software only, not tested on a Pi |
| Detection | OpenCV Haar cascade | OpenCV Haar cascade |
| Recognition | `face_recognition` (dlib embeddings) | OpenCV LBPH with a distance threshold |
| Notification | Mailgun email with photo | SMTP email (credentials from environment variables), plus a cooldown per visitor |
| Evaluation | One known/unknown accuracy test (91.4%) | A benchmark harness: face size, histogram equalization, simulated lighting, unknown-visitor threshold |

The lighting and latency experiments below are **new work in this recreation**; the original report did not include them.

## Pipeline
```
camera frame -> Haar face detector -> normalize (resize [+ histogram equalization])
             -> LBPH recognizer -> match distance <= threshold ? name : "unknown visitor"
             -> rate-limited email (cooldown per visitor)
```
Why these classic methods? They run in milliseconds on a Raspberry Pi CPU, which makes the
accuracy-vs-latency trade-off real instead of theoretical.

## Design
```
detector.py     FaceDetector protocol + Haar implementation
recognizer.py   LBPH wrapper with an 'unknown' threshold
pipeline.py     Doorbell: detect -> recognize -> cooldown-throttled notify
notifier.py     Notifier protocol: Email (SMTP, secrets from env) / Print (dry-run)
preprocess.py   normalize_face, adjust_gamma (lighting simulation)
dataset.py      folder-per-person loader + deterministic train/test split
benchmark.py    the experiments (see below)
```
Detector, recognizer, notifier and clock are all injected, so the whole pipeline is unit-tested
with no camera and no SMTP server.

## Research questions (benchmark.py)
1. **How much accuracy do we give up for speed?** Face size 32 / 64 / 100 px vs accuracy and p50/p95 latency.
2. **Does histogram equalization buy robustness to lighting?** Test photos are brightened/darkened with
   gamma 0.4 / 1.0 / 2.5 while the enrolled photos stay normally lit.
3. **Where should the "unknown visitor" threshold sit?** Sweep it and report the true-accept vs
   false-accept rates for registered visitors vs held-out strangers.

### Reproduce
Get a public face dataset laid out as `dataset/<person>/<images>`, for example the AT&T "Database of Faces"
(40 people x 10 grayscale photos; free for research, see its page for terms). Then:
```bash
uv venv && uv pip install pytest numpy "opencv-contrib-python-headless<5"
.venv/bin/python -m pytest -q
.venv/bin/python -m doorbell.benchmark --dataset dataset/att_faces --device-label "<your device>"
```
This writes `results/results.md`. **Results are not included yet** because I have not run this on a real dataset;
I'd rather publish nothing than numbers I didn't measure. Run it on a Raspberry Pi too and
compare the latency columns.

### Run the doorbell
```bash
python -m doorbell run --gallery faces/ --dry-run          # prints alerts
export DOORBELL_SMTP_HOST=... DOORBELL_SMTP_USER=... DOORBELL_SMTP_PASSWORD=... DOORBELL_NOTIFY_TO=...
python -m doorbell run --gallery faces/                   # sends email
```
`faces/` has one sub-folder of photos per registered visitor.

## Limitations (honest list)
- This recreation was developed without Raspberry Pi hardware or a live camera; the camera loop is untested
  and benchmark numbers come from the machine named in `results/results.md`.
- LBPH + Haar are 2000s-era methods: weak on pose, occlusion and large lighting changes; no liveness check
  (a photo held up to the camera would fool it). Modern deep models (e.g. OpenCV's YuNet + SFace) would do better.
- Gamma is a crude proxy for real lighting changes (no directional shadows or color casts).
- The benchmark uses pre-cropped face images, so it measures recognition, not detection.
- Unit tests use synthetic images; they prove the plumbing, not real-world accuracy.

## Future work
- Add a YuNet + SFace pipeline and put it on the same accuracy-latency plot.
- Measure on real Raspberry Pi hardware, including power/thermal throttling.
- Add a liveness / anti-spoofing check.
