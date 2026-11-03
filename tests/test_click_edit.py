from fastapi.testclient import TestClient
from PIL import Image

import app.main as main
from app.mapper import Element
from tests.test_fixtures_and_studio import _synthetic_curtain_wall


def test_click_add_and_remove_updates_session(monkeypatch, tmp_path):
    img, els = _synthetic_curtain_wall()
    st = main.SessionState(id="t", folder=tmp_path, img=img, els=els, night_base=None, building_mask=None, rig_options={})
    main.SESSIONS["t"] = st

    async def fake_sam(img, x, y, label, eid):  # SAM point call mocked
        m = Image.new("L", img.size, 0)
        m.paste(255, (x - 20, y - 20, x + 20, y + 20))
        return Element(eid, label, "other", (x - 20, y - 20, x + 20, y + 20), m)

    monkeypatch.setattr(main, "element_at_point", fake_sam)
    c = TestClient(main.app)
    r = c.post("/api/sessions/t/elements", json={"action": "add", "x": 180, "y": 300, "label": "mid band"})
    assert r.status_code == 200 and [e["id"] for e in r.json()["elements"]] == [1, 2, 3]
    assert st.els[-1].label == "mid band"  # stored map is what concepts/generate read
    assert r.json()["rig_options"]
    r = c.post("/api/sessions/t/elements", json={"action": "remove", "element_id": 1})
    assert [e["id"] for e in r.json()["elements"]] == [2, 3] and [e.id for e in st.els] == [2, 3]
