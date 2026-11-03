# Pipeline metrics

Reproduce: `uv run python scripts/metrics.py <image> [<image> ...]` (live API calls; one JSON line per image).
Single run per image, brief: "Slow pearlescent drift rising from podium to crown, resolving into a calm crown glow", 16 s show, recommended fixture rig.

## Coverage: 2 photos measured, 1 pending

| Facade | Status |
|---|---|
| A (daytime, 500x650) | measured |
| B (dusk, different viewpoint, 1200x1800) | measured |
| C | **pending** - no third distinct photo exists in `data/` or `out/sessions/`. Facades A and B look like the same tower; every other session image is a byte-identical copy or an upscale of one of these two. |

No numbers are estimated or extrapolated for C.

## Results

| Metric | Facade A | Facade B |
|---|---|---|
| F1 map_facade (s) | 58.6 | 53.9 |
| Fixture-rig survey (s) | 2.5 | 6.0 |
| F2a concept suggestions (s) | 51.2 | 24.9 |
| F2b show plan (s) | 12.1 | 19.4 |
| F5 render still + readability check (s) | 7.9 | 7.5 |
| Elements found | 9 | 6 |
| Elements with missing mask / tiny mask (<0.2% of image) | 0 / 0 | 0 / 0 |
| Fixtures (recommended rig) | 2330 | 1785 |
| Plan grounding (all element ids real) | pass | pass |
| Re-asks needed | 0 | 0 |
| Readability flags raised | 0 | 0 |
| Flags fixed on re-check | n/a (none raised) | n/a (none raised) |

## Misses and caveats

- **Slow stages.** F1 mapping takes ~54-59 s per photo (Muse Spark proposal plus several SAM 3.1 calls) and F2a concept suggestion 25-51 s. A full F1+F2+F5 pass is ~2.5 min, too slow for interactive use without caching.
- **F2a variance.** Concepts took 51 s vs 25 s on similar inputs; with n=1 per image we cannot say whether this is image-dependent or API latency.
- **Mask quality is not measured.** We only check that masks exist and are not tiny. There is no ground-truth IoU, so "0 bad masks" means "none obviously empty", not "all accurate". Facade B found only 6 elements (no fins, setback); at dusk the dark glass mass is probably under-segmented. Not verified visually in this run.
- **Validation and fix loop not exercised.** The planner passed grounding on the first attempt for both photos, so the re-ask path is untested here (0 re-asks is a measured 0, not evidence the retry works). The brief asks for no text, so the readability checker raised nothing, and the "flags raised vs. fixed" measurement is empty. The fix path (`apply_readability_fix`) needs a text-heavy brief to be measured.
- **F5 scope.** The check ran on one rendered still at 56% of the show (not every step), with the model critique on. The re-check after a fix would use deterministic checks only.
- **Tooling miss.** The first run crashed on image B (cached async client broken across separate `asyncio.run` calls) and an earlier attempt crashed serialising numpy ints. Both are fixed in `scripts/metrics.py`; the crashed attempts cost API credits and are not included above.
- Small sample (n=2, probably one building): treat these as indicative, not statistics.
