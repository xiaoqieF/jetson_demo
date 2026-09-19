import colorsys
import math
from typing import Any, Optional

import cv2
import numpy as np


def class_color(class_id: int) -> tuple[int, int, int]:
    hue = ((int(class_id) * 0.61803398875) % 1.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(blue * 255), int(green * 255), int(red * 255)


def map_and_clip_bbox(instance: Any, source_width: int, source_height: int,
                      target_width: int, target_height: int) -> Optional[tuple[int, int, int, int]]:
    if min(source_width, source_height, target_width, target_height) <= 0:
        return None
    values = (instance.x_min, instance.y_min, instance.x_max, instance.y_max)
    if not all(math.isfinite(float(value)) for value in values):
        return None
    scale_x, scale_y = target_width / source_width, target_height / source_height
    x1 = max(0, min(target_width, int(math.floor(float(instance.x_min) * scale_x))))
    y1 = max(0, min(target_height, int(math.floor(float(instance.y_min) * scale_y))))
    x2 = max(0, min(target_width, int(math.ceil(float(instance.x_max) * scale_x))))
    y2 = max(0, min(target_height, int(math.ceil(float(instance.y_max) * scale_y))))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def apply_instance_mask(image: np.ndarray, instance: Any, source_width: int,
                        source_height: int, color: tuple[int, int, int], alpha: float) -> bool:
    mask_width, mask_height = int(instance.mask_width), int(instance.mask_height)
    raw = np.asarray(instance.mask, dtype=np.uint8)
    if mask_width <= 0 or mask_height <= 0 or raw.size != mask_width * mask_height:
        return False
    image_height, image_width = image.shape[:2]
    if min(source_width, source_height, image_width, image_height) <= 0:
        return False
    source_x1, source_y1 = int(instance.mask_x), int(instance.mask_y)
    source_x2, source_y2 = source_x1 + mask_width, source_y1 + mask_height
    scale_x, scale_y = image_width / source_width, image_height / source_height
    full_x1, full_y1 = int(round(source_x1 * scale_x)), int(round(source_y1 * scale_y))
    full_x2, full_y2 = int(round(source_x2 * scale_x)), int(round(source_y2 * scale_y))
    target_width, target_height = full_x2 - full_x1, full_y2 - full_y1
    if target_width <= 0 or target_height <= 0:
        return False
    resized = cv2.resize(raw.reshape(mask_height, mask_width),
                         (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    clip_x1, clip_y1 = max(0, full_x1), max(0, full_y1)
    clip_x2, clip_y2 = min(image_width, full_x2), min(image_height, full_y2)
    if clip_x2 <= clip_x1 or clip_y2 <= clip_y1:
        return False
    mask_x1, mask_y1 = clip_x1 - full_x1, clip_y1 - full_y1
    mask_x2, mask_y2 = mask_x1 + clip_x2 - clip_x1, mask_y1 + clip_y2 - clip_y1
    selected = resized[mask_y1:mask_y2, mask_x1:mask_x2] != 0
    if not np.any(selected):
        return True
    roi = image[clip_y1:clip_y2, clip_x1:clip_x2]
    overlay = np.empty_like(roi)
    overlay[:] = color
    cv2.addWeighted(roi, 1.0 - alpha, overlay, alpha, 0.0, dst=overlay)
    roi[selected] = overlay[selected]
    return True


def draw_overlay(image: np.ndarray, result: Any, show_mask: bool = True,
                 show_boxes: bool = True, mask_alpha: float = 0.4) -> np.ndarray:
    output = image.copy()
    source_width, source_height = int(result.image_width), int(result.image_height)
    alpha = max(0.0, min(1.0, float(mask_alpha)))
    for instance in result.instances:
        color = class_color(instance.class_id)
        if show_mask:
            apply_instance_mask(output, instance, source_width, source_height, color, alpha)
        if not show_boxes:
            continue
        box = map_and_clip_bbox(instance, source_width, source_height,
                                output.shape[1], output.shape[0])
        if box is None:
            continue
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (x2 - 1, y2 - 1), color, 2)
        confidence = float(instance.confidence)
        confidence_text = f"{confidence * 100:.0f}%" if math.isfinite(confidence) else "?"
        label = f"{instance.class_name or instance.class_id} {confidence_text}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(0, y1 - text_height - baseline - 6)
        right = min(output.shape[1], x1 + text_width + 8)
        cv2.rectangle(output, (x1, top), (right, y1), color, cv2.FILLED)
        cv2.putText(output, label, (x1 + 4, max(text_height + 1, y1 - baseline - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return output
