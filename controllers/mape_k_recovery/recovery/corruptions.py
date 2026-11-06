"""Synthetic, sensor-fixed RGB and LiDAR faults for training and inference.

Partial RGB: rectangle/ellipse/polygon/blob/strip, random position, severity,
black/gray/colored contents. LiDAR: persistent close-range circular sector;
the same sector affects all 8 LiDAR frames. Full failure modes are explicit.
"""
from __future__ import annotations

import cv2
import numpy as np

SEVERITIES = (0.05, 0.10, 0.20, 0.35)
IMAGE_SHAPES = ("rectangle", "ellipse", "polygon", "blob", "strip")
IMAGE_CONTENTS = ("black", "gray", "color")
LIDAR_DISTANCES_M = (0.05, 0.10, 0.20, 0.30, 0.50)


def image_mask(height: int, width: int, rng: np.random.Generator,
               severity: float | None = None, shape: str | None = None):
    severity = float(rng.choice(SEVERITIES) if severity is None else severity)
    shape = str(rng.choice(IMAGE_SHAPES) if shape is None else shape)
    if not 0 < severity <= 1:
        raise ValueError("severity must be in (0,1]")
    target = height * width * severity
    aspect = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
    box_w = max(3, min(width, int(np.sqrt(target * aspect))))
    box_h = max(3, min(height, int(target / box_w)))
    cx, cy = int(rng.integers(width)), int(rng.integers(height))
    mask = np.zeros((height, width), np.uint8)
    if shape == "rectangle":
        mask[max(0, cy-box_h//2):min(height, cy+(box_h+1)//2),
             max(0, cx-box_w//2):min(width, cx+(box_w+1)//2)] = 1
    elif shape == "ellipse":
        cv2.ellipse(mask, (cx, cy), (box_w//2, box_h//2),
                    float(rng.uniform(0, 180)), 0, 360, 1, -1)
    elif shape in ("polygon", "blob"):
        angles = np.sort(rng.uniform(0, 2*np.pi, size=8 if shape == "polygon" else 14))
        r = rng.uniform(0.6 if shape == "polygon" else 0.3, 1, size=len(angles))
        pts = np.stack([cx + np.cos(angles)*r*box_w/2,
                        cy + np.sin(angles)*r*box_h/2], axis=1).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
        if shape == "blob":
            mask = (cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0) > .2).astype(np.uint8)
    elif shape == "strip":
        length = min(max(height, width), max(8, int(target**.5 * 3)))
        thick = max(2, int(target/length))
        box = cv2.boxPoints(((float(cx), float(cy)), (float(length), float(thick)),
                             float(rng.uniform(0, 180))))
        cv2.fillPoly(mask, [box.astype(np.int32)], 1)
    else:
        raise ValueError(shape)
    # Resize the occupied geometry to approach the chosen area, retaining placement.
    for _ in range(6):
        area = int(mask.sum())
        if area <= 0:
            mask[cy, cx] = 1
            continue
        if abs(area-target)/target < 0.1:
            break
        ys, xs = np.where(mask)
        crop = mask[ys.min():ys.max()+1, xs.min():xs.max()+1]
        scale = np.sqrt(target / area)
        nw = min(width, max(1, round(crop.shape[1]*scale)))
        nh = min(height, max(1, round(crop.shape[0]*scale)))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
        x0, y0 = int(np.clip(cx-nw//2, 0, width-nw)), int(np.clip(cy-nh//2, 0, height-nh))
        mask.fill(0)
        mask[y0:y0+nh, x0:x0+nw] = resized
    return mask.astype(bool), dict(shape=shape, target_fraction=severity,
                                    actual_fraction=float(mask.mean()))


def corrupt_image(image: np.ndarray, rng: np.random.Generator,
                  severity: float | None = None, full: bool = False):
    image = np.asarray(image, dtype=np.uint8)
    if image.shape[-1] != 3:
        raise ValueError("Expected HWC 3-channel camera observation")
    result = image.copy()
    if full:
        result.fill(0)
        return result, {"kind": "full_image", "fraction": 1.0}
    mask, spec = image_mask(image.shape[0], image.shape[1], rng, severity=severity)
    content = str(rng.choice(IMAGE_CONTENTS))
    color = (0, 0, 0) if content == "black" else (127, 127, 127) if content == "gray" else tuple(int(c) for c in rng.integers(0, 256, size=3))
    result[mask] = color
    return result, {"kind": "partial_image", "content": content, "color": list(color), **spec}


def corrupt_lidar(lidar: np.ndarray, rng: np.random.Generator,
                  severity: float | None = None, full: bool = False):
    x = np.asarray(lidar, dtype=np.float32)
    if x.shape != (8, 1200):
        raise ValueError(f"Expected (8,1200) padded stacked LiDAR, got {x.shape}")
    y = x.copy()
    if full:
        y.fill(1.0)
        return y, {"kind": "full_lidar", "fraction": 1.0}
    severity = float(rng.choice(SEVERITIES) if severity is None else severity)
    start = int(rng.integers(1080))
    width = max(1, round(1080*severity))
    distance = float(rng.choice(LIDAR_DISTANCES_M))
    value = 1.0 - distance/12.5
    indices = (start + np.arange(width)) % 1080
    # operate in raw angular sensor coordinates; reconstruct circular 60-sample padding
    raw = y[:, 60:1140].copy()
    raw[:, indices] = np.clip(value + rng.normal(0, .001, size=(8, width)), 0, 1)
    y = np.concatenate((raw[:, -60:], raw, raw[:, :60]), axis=1)
    return y, {"kind": "partial_lidar", "fraction": severity, "start": start,
               "width": width, "distance_m": distance, "value": value}


def apply_condition(observation: dict, condition: str, rng: np.random.Generator):
    """Copy and corrupt only targeted modality; world-info follows LiDAR dependency."""
    valid = {"normal", "partial_image", "full_image", "partial_lidar", "full_lidar", "partial_both", "full_both"}
    if condition not in valid:
        raise ValueError(f"Unknown condition: {condition}")
    obs = {k: np.asarray(v).copy() for k, v in observation.items()}
    descriptors = {}
    if condition.endswith("image") or condition.endswith("both"):
        obs["image"], descriptors["image"] = corrupt_image(obs["image"], rng, full=condition.startswith("full"))
    if condition.endswith("lidar") or condition.endswith("both"):
        obs["lidar"], descriptors["lidar"] = corrupt_lidar(obs["lidar"], rng, full=condition.startswith("full"))
        if "world_info" in obs:
            if condition.startswith("full"):
                obs["world_info"] = np.zeros_like(obs["world_info"])
            else:
                obs["world_info"] = np.clip(obs["world_info"] + rng.normal(0, .02, size=3), 0, 1)
    return obs, descriptors
