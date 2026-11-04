import numpy as np

from app.survey import Survey, apply, elevations, homography, pick_elevation, quad_from_mask


def test_homography_maps_corners():
    src = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], float)
    dst = np.array([[5, 5], [15, 7], [14, 30], [4, 28]], float)
    x, y = apply(homography(src, dst), src[:, 0], src[:, 1])
    assert np.allclose(np.stack([x, y], 1), dst, atol=1e-6)


def test_elevations_split_at_gaps_and_skip_short_bands():
    cx = np.r_[np.full(10, 100.0), np.full(10, 600.0), np.full(10, 1100.0)]
    cy = np.r_[np.linspace(0, 900, 10), np.linspace(0, 900, 10), np.linspace(900, 950, 10)]  # 3rd = podium band
    sv = Survey(cx, cy, 10, 20, (1200, 1000))
    r = elevations(sv)
    assert len(r) == 2
    assert pick_elevation(r, 0.1) in (0, 1)


def test_quad_from_mask():
    m = np.zeros((100, 60), bool); m[10:90, 20:40] = True
    q = quad_from_mask(m)
    assert q[0].tolist() == [20, 10] and q[2].tolist() == [39, 89]


def test_silhouette_edges_of_a_rectangle():
    from app.facets import silhouette_edges
    m = np.zeros((100, 60), bool); m[10:90, 20:40] = True
    yt, yb, L, R = silhouette_edges(m)
    assert (yt, yb) == (10, 89) and abs(L(50) - 20) < 1e-6 and abs(R(50) - 39) < 1e-6


def test_survey_rig_covers_both_faces_and_podium():
    """Two tall elevations + a podium band must land on a 2-face tower plus its annex (coverage regression)."""
    from PIL import Image
    from app.fixtures import _survey_rig_option
    from app.mapper import Element
    gx, gy = np.meshgrid(np.arange(0, 400, 20), np.arange(0, 1000, 40))
    tall = lambda x0: (gx.ravel() + x0, gy.ravel() + 0)
    podx, pody = np.meshgrid(np.arange(0, 600, 20), np.arange(0, 100, 40))
    cx = np.r_[tall(100)[0], tall(700)[0], podx.ravel() + 1300] + 0.0
    cy = np.r_[tall(100)[1], tall(700)[1], pody.ravel() + 900] + 0.0
    sv = Survey(cx, cy, 10.0, 20.0, (2000, 1000))
    H, W = 800, 700
    tower = np.zeros((H, W), bool); tower[50:700, 100:300] = True
    bmask = tower.copy(); bmask[600:700, 300:600] = True
    el = Element(1, "tower", "tower", (100, 50, 300, 700), Image.fromarray(tower.astype(np.uint8) * 255))
    o = _survey_rig_option(sv, [el], {1: tower}, bmask, W, H)
    cells = {(int(y) // 64, int(x) // 64) for x, y in zip(o.xs, o.ys)}
    want = {(j, i) for j in range(0, H // 64) for i in range(0, W // 64) if bmask[j * 64:(j + 1) * 64, i * 64:(i + 1) * 64].mean() > 0.5}
    assert len(cells & want) / len(want) >= 0.9
    assert (o.xs > 300).any() and (o.xs > 100).sum() > 0.5 * len(o.xs)  # annex got fixtures
