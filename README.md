# facade-mapper-intel

Reads a building's architecture from one photo and makes show content that fits it.

- **SAM 3.1** maps the facade's elements (fins, glass bands, setbacks, crown).
- **Muse Spark** plans the show against those elements and checks readability.
- **Muse Image** makes per-element visuals.
- Our own code moves content along element outlines.

## Built during the event

Everything in this repository was written 26 Oct – 4 Nov 2026 for the Meta AI Developer Hackathon (Startup track). Commit history is the evidence; see [CHANGELOG.md](CHANGELOG.md).

- **F1 Element map**: facade elements (fins, glass bands, setbacks, crown) as masks from one photo (`app/mapper.py`, `POST /map`)
- **F2 Show plan**: timed plan whose steps reference real element ids, plus concept suggestions (`app/planner.py`, `POST /plan`)
- **F3 Visuals**: per-concept light-art textures and 360-degree synthesizer (`app/visuals.py`)
- **F4 Motion**: element-driven motion and fixture rendering (`app/motion.py`, `app/fixtures.py`)
- **F5 Readability check**: fixture-pitch audit and vision review, with auto-fix (`app/checker.py`)
- **Studio UI**: browser interface and session API (`GET /`, `/api/*`)
- **Four-side stitcher**: combines North, East, South, West photos into one 360-degree facade map (`app/stitcher.py`)

## Existing baseline (not submitted)

A private Facade Mapper app existed before the event and is not part of this repository:

- zone drawing
- video generation
- export
- the `facade_videogen` facet/homography contract (per-face canvas-rect to photo-quad mapping)

## How the two connect

This repo is a standalone service. The existing app calls it over HTTP (`POST /map`, `POST /plan`, and `/api/*`). No baseline code is included here.

## Which model runs what

| Feature | Model | Notes |
|---|---|---|
| F1 Element map | Muse Spark 1.3 + SAM 3.1 | Spark proposes element boxes; SAM 3.1 outlines them |
| F2 Plan, concepts | Muse Spark 1.3 | JSON plan; invalid plans re-asked once |
| F3 Visuals | Muse Image 1.0 | Falls back to a procedural texture when the image call fails; the response reports which source was used |
| F4 Motion | None (our own code) | NumPy/ffmpeg rendering along element masks |
| F5 Readability check | Muse Spark 1.3 | Vision review plus a deterministic pitch audit |
| Stitcher | SAM 3.1 + Muse Spark 1.3 | Segments each side |

## Results

See [METRICS.md](METRICS.md).

## Roadmap (from user feedback)

Requested by a lighting designer after seeing the demo; not built in the event window.

- Adjustable mask size and width (grow/shrink and resize per element).
- Auto-generated masks that can be moved and reshaped by hand, not only added or deleted.
- Linear (strip) and pixel-dot fixture types, selectable per element, so the designer controls what the rig is made of.

## Run

```bash
cp .env.example .env   # add MODEL_API_KEY from https://dev.meta.ai
uv sync
uv run uvicorn app.main:app --reload   # open http://127.0.0.1:8000 for Facade Mapper Studio
uv run python scripts/smoke.py [facade.jpg]
uv run pytest
```

Endpoints (`app/main.py`): `GET /health`, `GET /`, `POST /map`, `POST /plan`, `GET /api/presets`, `POST /api/analyze`, `POST /api/analyze-stitch`, `POST /api/concepts`, `POST /api/generate`, `POST /api/fix-readability`, `GET /assets/{session_id}/{filename}`.

### Demo sign-in
Set `DEMO_PASSWORD` (and optionally `DEMO_USERS`, default `judge,designer`) in `.env` to gate the Studio and API behind a login. With `DEMO_PASSWORD` unset, auth is off (local dev and tests). The landing page is at `/`, the Studio at `/studio`.

### Studio Workflow (`http://127.0.0.1:8000`)
1. **Upload 1 viewpoint photo OR Stitch All 4 Sides (North · East · South · West)**:
   - **Single Photo (1–2 Sides)**: upload one photo (plus an optional 2D UV fixture survey mask); `muse-spark-1.3` + `sam-3.1` map the architectural elements and trace real mullions, helical fins, crown, setbacks, and floor slabs to suggest **3+ realistic fixture rig options**.
   - **Stitch All 4 Sides (360° Full Facade Map)**: upload all 4 building elevations (`0°–90°`, `90°–180°`, `180°–270°`, `270°–360°`); `app/stitcher.py` segments all 4 sides concurrently, normalizes crown-to-base scale, harmonizes twilight sky exposure, and stitches discrete mullion/fin fixtures across all 4 quadrants into one unified 360° facade map.
2. **Choose a fixture layout** — `muse-spark-1.3` proposes **3 tailored lighting concepts** grounded in the mapped zones and fixture count, with instant per-fixture concept mockups.
3. **Generate 360° show & export media-server deliverables** — authors a seamless 360° wrap-around show across all building elevations (`muse-image-1.0` silk plates + helical vortex, comet stream, diamond lattice, and champagne shimmer shaders), renders discrete physical LED bars (zero glass wash), runs the `muse-spark-1.3` readability check + 1-click fix, and exports both the on-building video and the pixel-registered whole-facade media-server deliverable + UV mask.

## Sample photos and credits
`app/static/samples/index_tower.jpg` ("The Index Tower" by Aidas U., CC BY 3.0, https://commons.wikimedia.org/wiki/File:The_Index_Tower_-_panoramio.jpg, downscaled) is the bundled public sample, offered as preset `index_tower`. The Wasl presets only appear when the private `data/wasl.png` exists. The four-sides demo reuses one photo and is not real elevations.

## Deploy

Run it locally (see Run above) or with the `Dockerfile` (Python 3.12 + ffmpeg, serves on `$PORT`, needs `MODEL_API_KEY`).

Memory: measured peaks on a 500x650 photo are about 650 MB for `/api/analyze` and 1.8 GB for `/api/generate`, so a 512 MB host (Render Starter) is too small and 2 GB is tight. Meta's API also rejected requests from Render's Frankfurt region (404 "model not found") but accepted Virginia and local runs. For a shareable demo URL, run locally behind a tunnel, e.g. `cloudflared tunnel --url http://localhost:8000`.

License: Apache-2.0
