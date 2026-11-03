# facade-mapper-intel

Reads a building's architecture from one photo and makes show content that fits it.

- **SAM 3.1** maps the facade's elements (fins, glass bands, setbacks, crown).
- **Muse Spark** plans the show against those elements and checks readability.
- **Muse Image** makes per-element visuals.
- Our own code moves content along element outlines.

## Built during the event

Everything in this repository was written 26 Oct – 4 Nov 2026 for the Meta AI Developer Hackathon (Startup track).
Commit history is the evidence; see [CHANGELOG.md](CHANGELOG.md).

> A pre-existing Facade Mapper app exists separately (private) and is **not** submitted.
> This repo runs as a standalone service with its own interface; the existing app calls it.

## Run

```bash
cp .env.example .env   # add MODEL_API_KEY from https://dev.meta.ai
uv sync
uv run uvicorn app.main:app --reload   # open http://127.0.0.1:8000 for Facade Mapper Studio
uv run python scripts/smoke.py [facade.jpg]
uv run pytest
```

### Studio Workflow (`http://127.0.0.1:8000`)
1. **Upload 1 viewpoint photo OR Stitch All 4 Sides (North · East · South · West)**:
   - **Single Photo (1–2 Sides)**: upload one photo (plus an optional 2D UV fixture survey mask); `muse-spark-1.3` + `sam-3.1` map the architectural elements and trace real mullions, helical fins, crown, setbacks, and floor slabs to suggest **3+ realistic fixture rig options**.
   - **Stitch All 4 Sides (360° Full Facade Map)**: upload all 4 building elevations (`0°–90°`, `90°–180°`, `180°–270°`, `270°–360°`); `app/stitcher.py` segments all 4 sides concurrently, normalizes crown-to-base scale, harmonizes twilight sky exposure, and stitches discrete mullion/fin fixtures across all 4 quadrants into one unified 360° facade map.
2. **Choose a fixture layout** — `muse-spark-1.3` proposes **3 tailored lighting concepts** grounded in the mapped zones and fixture count, with instant per-fixture concept mockups.
3. **Generate 360° show & export media-server deliverables** — authors a seamless 360° wrap-around show across all building elevations (`muse-image-1.0` silk plates + helical vortex, comet stream, diamond lattice, and champagne shimmer shaders), renders discrete physical LED bars (zero glass wash), runs the `muse-spark-1.3` readability check + 1-click fix, and exports both the on-building video and the pixel-registered whole-facade media-server deliverable + UV mask.

Hosting: Render web service (long-running Python process; no function timeouts for model calls).

License: Apache-2.0
