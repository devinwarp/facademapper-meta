import numpy as np
from types import SimpleNamespace as NS

from app.mapper import place_tile


def test_downscaled_sam_tile_is_stretched_to_its_bounds():
    # SAM sent a 4x8 tile (w x h) for an object whose bounds are 8x16 px at (left=10, top=20) of a 100x60 image
    tile = np.zeros((8, 4), bool); tile[:, 2:] = True  # right half set
    full = place_tile(tile, NS(left=10, top=20, right=18, bottom=36), 60, 100)
    assert full.sum() == 4 * 16 and full[20:36, 14:18].all() and not full[:, :14].any()


def test_tile_clipped_at_image_edge_and_native_size_untouched():
    full = place_tile(np.ones((10, 10), bool), NS(left=95, top=0, right=105, bottom=10), 20, 100)
    assert full.sum() == 5 * 10
