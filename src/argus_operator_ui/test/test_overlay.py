from types import SimpleNamespace

import numpy as np

from argus_operator_ui.overlay import apply_instance_mask, draw_overlay, map_and_clip_bbox


def instance(**overrides):
    values = dict(class_id=1, class_name="person", confidence=0.9,
                  x_min=10.0, y_min=10.0, x_max=50.0, y_max=40.0,
                  mask_x=0, mask_y=0, mask_width=0, mask_height=0, mask=[])
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bbox_scaling_for_different_image_size():
    assert map_and_clip_bbox(instance(), 100, 50, 200, 100) == (20, 20, 100, 80)


def test_bbox_clipping_and_invalid_values():
    assert map_and_clip_bbox(instance(x_min=-10, y_min=-5, x_max=120, y_max=60),
                             100, 50, 200, 100) == (0, 0, 200, 100)
    assert map_and_clip_bbox(instance(x_min=float("nan")), 100, 50, 200, 100) is None
    assert map_and_clip_bbox(instance(x_min=20, x_max=10), 100, 50, 200, 100) is None


def test_empty_and_bad_length_masks_are_ignored():
    image = np.zeros((20, 20, 3), np.uint8)
    assert not apply_instance_mask(image, instance(), 20, 20, (0, 255, 0), 0.4)
    bad = instance(mask_width=2, mask_height=2, mask=[255, 0, 255])
    assert not apply_instance_mask(image, bad, 20, 20, (0, 255, 0), 0.4)
    assert not image.any()


def test_mask_roi_clips_without_crashing():
    image = np.zeros((10, 10, 3), np.uint8)
    mask = instance(mask_x=8, mask_y=8, mask_width=5, mask_height=5, mask=[255] * 25)
    assert apply_instance_mask(image, mask, 10, 10, (0, 255, 0), 0.4)
    assert image[8:, 8:, 1].all()
    assert not image[:8].any()


def test_draw_overlay_accepts_empty_mask():
    image = np.zeros((50, 100, 3), np.uint8)
    result = SimpleNamespace(image_width=100, image_height=50, instances=[instance()])
    output = draw_overlay(image, result)
    assert output.shape == image.shape
    assert output.any()
