# Pipeline metrics

Reproduce: `uv run python scripts/metrics.py <image> [<image> ...]` (live API calls; one JSON line per image).
Single run per image, brief: "Slow pearlescent drift rising from podium to crown, resolving into a calm crown glow", 16 s show, recommended fixture rig. Same brief and settings as the first measurement, so numbers are comparable.

## Coverage: 3 photos measured, n=3, one run each

| Facade | Photo |
|---|---|
| A | daytime, 500x650 |
| B | night, 1492x2240, different viewpoint (see caveat: appears to be the same tower as A) |
| C | Index Tower, Dubai, daytime, 1680x2240. Public sample, "The Index Tower" by Aidas U., CC BY 3.0 (https://commons.wikimedia.org/wiki/File:The_Index_Tower_-_panoramio.jpg) |

Only C is a verifiably different building. A and B both show a twisting, tapering tower with a low podium block beside it; visually they look like the same building (day vs. night, different angle). We did not confirm this against any source, so call it "likely the same building", not fact. Effective distinct buildings: 2, not 3.

## Results

| Metric | Facade A | Facade B | Facade C |
|---|---|---|---|
| F1 map_facade (s) | 60.2 | 75.1 | 101.6 |
| Fixture-rig survey (s) | 3.0 | 6.8 | 4.7 |
| F2a concept suggestions (s) | 37.0 | 37.7 | 32.9 |
| F2b show plan (s) | 21.1 | 17.3 | 22.9 |
| F5 render still + readability check (s) | 9.5 | 15.6 | 15.2 |
| Elements found | 9 | 8 | 9 |
| Elements with missing mask / tiny mask (<0.2% of image) | 0 / 0 | 0 / 1 | 0 / 0 |
| Fixtures (recommended rig) | 2437 | 1787 | 2384 |
| Plan grounding (all element ids real) | pass | pass | pass |
| Planner model calls / re-asks | 1 / 0 | 2 / 1 | 3 / 2 |
| Readability flags raised | 0 | 0 | 0 |
| Flags fixed on re-check | n/a (none raised) | n/a (none raised) | n/a (none raised) |
| Script errors | none | none | none |

Element kinds found: A and C: band, crown, fins, glass, podium, setback, tower. B: same minus setback.

## What changed since the first measurement

Two fixes landed after the first run: SAM mask tiles are now resized to their bounds before pasting (`app/mapper.py` `place_tile`), and the survey rig now maps two elevations onto both faces plus podium.

Facade A, old (first run) vs. new:

| Metric | Old | New |
|---|---|---|
| Elements found | 9 | 9 |
| Fixtures (recommended rig) | 2330 | 2437 |
| F1 map_facade (s) | 58.6 | 60.2 |
| Rig survey (s) | 2.5 | 3.0 |
| F2a concepts (s) | 51.2 | 37.0 |
| F2b plan (s) | 12.1 | 21.1 |
| F5 check (s) | 7.9 | 9.5 |
| Re-asks | 0 | 0 |

- A's element count did not change (9 and 9); fixtures rose 4.6% (2330 to 2437). We did not isolate which fix caused that, and mask-model output is itself non-deterministic, so some of the change may be run-to-run variation.
- Timing differences on A (F2a -14 s, F2b +9 s) are within what we saw as API latency variance; n=1 each, so not attributable to either fix.
- Old Facade B (1200x1800 dusk photo) is not the same file as the current B (1492x2240 night photo), so no old-vs-new comparison is given for B. The old "Facade C pending" is now resolved with the Index Tower sample.

## Misses and caveats

- **Slow stages.** F1 mapping took 60-102 s per photo (Muse Spark proposal plus several SAM 3.1 calls); it got slower with image size (500x650: 60 s; 1492x2240: 75 s; 1680x2240: 102 s, with n=3 we cannot separate size from API latency). A full F1+F2+F5 pass is ~2.2 to 3.0 min per photo, too slow for interactive use without caching.
- **Re-asks happened this time.** The planner needed 1 re-ask on B and 2 on C (calls minus 1), but all final plans passed grounding. This exercises the F2 re-ask path for the first time, but we did not log why the first attempts failed, so we cannot say what the re-asks corrected. It also adds ~17-23 s per extra call to those F2b times.
- **Tiny mask on B.** One of 8 elements on B has a mask under 0.2% of the image. We did not view which element it is or whether it is a real small feature or a segmentation failure. B has no setback element, probably missed on the night photo.
- **Mask accuracy is not measured.** We only check that masks exist and are not tiny. There is no ground-truth IoU, so "0 missing masks" means "none obviously empty", not "all accurate". We did not visually inspect the masks in this run, including to verify the mask-pasting fix.
- **F5 fix path still not exercised.** The brief asks for no text, so the readability checker raised 0 flags on all three photos; "flags raised vs. fixed" is empty and `apply_readability_fix` was never run. A text-heavy brief is needed to measure it.
- **F5 scope.** The check ran on one rendered still at 56% of the show, not every step, with the model critique on.
- **Sample size.** n=3 photos, one run each, about two distinct buildings. Treat as indicative, not statistics. Single-run timings vary with API latency.
