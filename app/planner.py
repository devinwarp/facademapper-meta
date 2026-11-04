"""F2: Architecture-aware concept suggestions & grounded show plans via Muse Spark (muse-spark-1.3).

1. `suggest_concepts()` proposes 3 distinct lighting concepts tailored to the building's mapped
   elements and the user's selected fixture rig option.
2. `make_plan()` writes a timed JSON show plan whose every step points to real element ids;
   invalid plans are rejected and re-asked once.
"""
from __future__ import annotations

import asyncio, json
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, ValidationError, model_validator

from app.mapper import Element, _client, _data_url

MOTIONS = [
    "climb",
    "descend",
    "sweep_left_to_right",
    "sweep_right_to_left",
    "pulse",
    "fill",
    "text",
    "logo",
    "shimmer",
    "hold",
    "aurora",
    "chase",
]


class ConceptOption(BaseModel):
    id: str
    title: str
    tagline: str = Field(description="1 sentence naming how light moves on this building's real elements")
    mood: str = Field(description="2-3 word style tag, e.g. 'Civic & Bold' or 'Architectural Gold'")
    palette: list[str] = Field(min_length=2, max_length=5, description="hex colours like #EF3340")
    brief: str = Field(description="detailed show direction naming the architectural zones and transitions")
    visual_prompt: str = Field(description="prompt for abstract luminous LED colour texture matching this concept")


class ConceptList(BaseModel):
    concepts: list[ConceptOption] = Field(min_length=3, max_length=4)


class Step(BaseModel):
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    element_ids: list[int] = Field(min_length=1)
    motion: Literal[tuple(MOTIONS)]  # type: ignore[valid-type]
    colors: list[str] = Field(description="hex colours like #00732F", min_length=1)
    text: str | None = Field(default=None, description="exact words to show, for motion=text")
    description: str = Field(description="one sentence naming the elements, e.g. 'the wave climbs the tower'")


class Plan(BaseModel):
    title: str
    duration_s: float = Field(gt=0)
    steps: list[Step] = Field(min_length=1)

    @model_validator(mode="after")
    def _times(self):
        for s in self.steps:
            if s.end_s <= s.start_s or s.end_s > self.duration_s + 1e-6:
                raise ValueError(f"step {s.start_s}-{s.end_s}s is outside 0-{self.duration_s}s")
        return self


def check_grounding(plan: Plan, valid_ids: set[int]) -> list[str]:
    """Every step must name real element ids. Returns problems (empty = grounded)."""
    return [
        f"step {i} ({s.description!r}) uses unknown element ids {sorted(set(s.element_ids) - valid_ids)}"
        for i, s in enumerate(plan.steps)
        if not set(s.element_ids) <= valid_ids
    ]


CONCEPT_SYSTEM = (
    "You are an executive architectural media-facade creative director following 3S Lighting's 'Museum of Light' philosophy "
    "(3S Lighting's landmark-tower standard): 'The tower is a sculpture first and a screen never; emerge, do not overlay; read at the pitch.' "
    "The building carries discrete linear LED fixtures across its 360° elevations; the photograph shows the camera's visible side(s).\n"
    "Avoid garish multi-colour neon stripes or busy graphics. Propose 3 distinct, museum-grade architectural lighting concepts tailored to THIS building's geometry and fixture rig:\n"
    "1. Pearlescent Drift & Spine Gathering (restrained azure `#3A7BD5`, soft lilac `#B58FD6`, and warm silver `#E6E9F0` drifting diagonally across the fins and gathering into a luminous vertical spine).\n"
    "2. Suhail — Rising Star & Nocturnal Cooling (desert amber `#F0B36A` and dusky rose `#C76A7A` cooling into twilight indigo `#3B2F7A` and deep midnight `#15204F` as a warm-white star `#FFF4E8` ascends the structural spine, leaving sparse diamond dew sparkles).\n"
    "3. Night Bloom & Braided Silk Strands (a calm pearlescent veil `#CFD6E6` / `#8E9BBF` from which a sculptural warm coral `#FF6F61` & ivory `#FFF4EE` botanical bloom unfolds at mid-tower, resolving into intertwined cosine silk strands).\n"
    "Rules:\n"
    "- Every concept's `brief` MUST treat the facade from base to crown as one cohesive architectural sculpture, naming how light honours structural spines, recessed glass negative space, setbacks, and crown.\n"
    "- `visual_prompt` MUST describe ONLY a flat 2D minimalist low-frequency pearlescent colour & silk light gradient (zero mention of buildings, towers, windows, sky, or text).\n"
    "- Do NOT propose `text` on the facade unless explicitly asked; think in pure kinetic architectural light.\n"
    "- Use restrained, curated hex palettes (3 to 4 harmonious colours per concept; never `#000000`)."
)


async def suggest_concepts(
    img: Image.Image,
    els: list[Element],
    rig_summary: dict | None = None,
) -> list[ConceptOption]:
    """F2a: Muse Spark proposes 3 distinct lighting concepts grounded in the mapped building + chosen fixture rig."""
    elist = [{"id": e.id, "label": e.label, "kind": e.kind, "box": e.box} for e in els]
    rig_txt = json.dumps(rig_summary) if rig_summary else "Standard mullion linear LED rig"
    r = await _client().chat.completions.create(
        model="muse-spark-1.3",
        messages=[
            {"role": "system", "content": CONCEPT_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": f"Selected fixture rig: {rig_txt}\nMapped architectural elements: {json.dumps(elist)}"},
                {"type": "image_url", "image_url": {"url": _data_url(img)}},
            ]},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "concepts", "schema": ConceptList.model_json_schema()}},
    )
    parsed = ConceptList.model_validate_json(r.choices[0].message.content)
    for i, c in enumerate(parsed.concepts, start=1):
        c.id = f"concept_{i}"
    return parsed.concepts


SYSTEM = (
    "You choreograph museum-grade architectural media-facade shows (3S Lighting's landmark-tower standard) across a 360° building lit by discrete linear LED fixtures. "
    "Principle: 'The tower is a sculpture first and a screen never; emerge, do not overlay.' "
    "The photo shows the visible side(s) of the building; the element list gives its mapped architectural zones.\n"
    "Rules for sculptural choreography:\n"
    "- UNIFIED FULL-FACADE SCULPTURE: Every primary movement (`climb`, `aurora`, `pulse`, `chase`, `shimmer`, `sweep_left_to_right`, `descend`) MUST list ALL main facade `element_ids` together so light moves continuously from podium to crown!\n"
    "- MEDITATIVE 3–4 ACT NARRATIVE ARC: Compose 3 to 4 unhurried, poetic movements over `duration_s` — e.g. Arrival (`aurora` pearlescent drift & spine gathering or `climb` Suhail rising star) -> Focal Emergence (`pulse` 5-lobed night bloom or `chase` braided silk strands) -> Resolution (`shimmer` nocturnal dew scintillation or `aurora` calm horizon hold).\n"
    "- LAYERED CROWN / SETBACK ACCENTS: Optionally add a gentle simultaneous accent step on `crown` or `setback` element_ids.\n"
    "- RESTRAINED PALETTES: Keep each step's `colors` faithful to the brief's curated 3–4 stop palette (never clashing rainbows, never `#000000`).\n"
    "- NO UNSOLICITED TEXT: NEVER use `motion='text'` unless the user's brief explicitly asks to display specific quoted words.\n"
    "- Use only element ids from the provided list and only these motions: " + ", ".join(MOTIONS) + "."
)


async def make_plan(
    img: Image.Image,
    els: list[Element],
    brief: str,
    duration_s: float = 16,
    elem_fixture_counts: dict[int, int] | None = None,
) -> Plan:
    elist = [
        {
            "id": e.id,
            "label": e.label,
            "kind": e.kind,
            "box": e.box,
            **({"fixtures": int(elem_fixture_counts.get(e.id, 0))} if elem_fixture_counts is not None else {}),
        }
        for e in els
    ]
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": f"Brief: {brief}\nDuration: {duration_s}s\nImage size: {img.width}x{img.height}\nElements: {json.dumps(elist)}"},
            {"type": "image_url", "image_url": {"url": _data_url(img)}},
        ]},
    ]
    valid = {e.id for e in els}
    problems: list[str] = []
    for _ in range(2):
        r = await _client().chat.completions.create(
            model="muse-spark-1.3",
            messages=msgs,
            response_format={"type": "json_schema", "json_schema": {"name": "plan", "schema": Plan.model_json_schema()}},
        )
        raw = r.choices[0].message.content
        try:
            plan = Plan.model_validate_json(raw)
            problems = check_grounding(plan, valid)
        except ValidationError as ex:
            plan, problems = None, [str(ex)[:600]]
        if not problems and plan is not None:
            return plan
        msgs += [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Fix these problems and return the full plan again:\n" + "\n".join(problems)},
        ]
    raise ValueError("plan failed validation twice: " + "; ".join(problems))


if __name__ == "__main__":  # python -m app.planner photo.png "brief"
    import sys
    from app.mapper import map_facade
    im = Image.open(sys.argv[1])
    els = asyncio.run(map_facade(im))
    plan = asyncio.run(make_plan(im, els, sys.argv[2]))
    print(plan.model_dump_json(indent=1))
