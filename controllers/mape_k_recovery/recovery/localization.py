"""Active scan and sensor-fixed image/LiDAR localization.

The localization functions are retained from the final recovery controller.
No simulator ground-truth fault coordinates are read for diagnosis.
"""
from __future__ import annotations
from typing import Any, Optional, Dict
import numpy as np
import cv2

def _latest_unpadded_lidar(lidar_obs: np.ndarray, padding: int = 60) -> np.ndarray:
    arr = np.asarray(lidar_obs, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[-1]
    if arr.ndim != 1:
        raise ValueError(f"Unexpected LiDAR scan shape: {arr.shape}")
    if arr.size > 2 * padding:
        return arr[padding:-padding].copy()
    return arr.copy()

def _circular_largest_true_run(mask: np.ndarray, min_width: int) -> Optional[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.size)
    if n == 0 or not mask.any():
        return None
    doubled = np.concatenate([mask, mask])
    best_start, best_len = None, 0
    i = 0
    while i < 2 * n:
        if not doubled[i]:
            i += 1
            continue
        j = i
        while j < 2 * n and doubled[j]:
            j += 1
        run_len = min(j - i, n)
        start = i % n
        if run_len > best_len:
            best_start, best_len = start, run_len
        i = j
    if best_start is None or best_len < min_width:
        return None
    return int(best_start), int(best_len)

def localize_image_region(frames: list[np.ndarray], args) -> Dict[str, Any]:
    if len(frames) < 2:
        raise RuntimeError("Need at least two image frames for image localization.")
    stack = np.stack([np.asarray(f)[..., :3].astype(np.float32) for f in frames], axis=0)
    h, w = stack.shape[1], stack.shape[2]

    pixel_range = stack.max(axis=0) - stack.min(axis=0)
    stable = np.all(pixel_range <= float(args.image_stability_tol), axis=2).astype(np.uint8)

    k = max(1, int(args.image_morph_kernel))
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        stable = cv2.morphologyEx(stable, cv2.MORPH_OPEN, kernel)
        stable = cv2.morphologyEx(stable, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(stable, connectivity=8)
    max_area = float(args.image_max_component_fraction) * h * w
    candidates = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < int(args.image_min_component_area):
            continue
        if area > max_area:
            continue
        candidates.append((int(area), int(x), int(y), int(bw), int(bh), idx))

    if not candidates:
        # Fallback: choose the largest non-background component, even if thresholds were too strict.
        for idx in range(1, num_labels):
            x, y, bw, bh, area = stats[idx]
            if area >= int(args.image_min_component_area):
                candidates.append((int(area), int(x), int(y), int(bw), int(bh), idx))
    if not candidates:
        raise RuntimeError("Image localization failed: no sufficiently stable connected component found.")

    area, x, y, bw, bh, idx = sorted(candidates, reverse=True)[0]
    component_mask = labels == idx
    first = stack[0]
    if component_mask.any():
        color = np.median(first[component_mask], axis=0)
    else:
        color = np.median(first[y:y + bh, x:x + bw].reshape(-1, 3), axis=0)
    color = [int(np.clip(round(float(c)), 0, 255)) for c in color]

    return {
        "x": x,
        "y": y,
        "width": bw,
        "height": bh,
        "color": color,
        "stable_area": area,
        "n_scan_frames": len(frames),
    }

def localize_lidar_region(scans: list[np.ndarray], args) -> Dict[str, Any]:
    if len(scans) < 2:
        raise RuntimeError("Need at least two LiDAR scans for LiDAR localization.")

    stack = np.stack([np.asarray(s, dtype=np.float32) for s in scans], axis=0)
    temporal_range = stack.max(axis=0) - stack.min(axis=0)
    temporal_mean = stack.mean(axis=0)

    def _summary(prefix: str) -> str:
        return (
            f"{prefix}: n_scans={len(scans)}, width={stack.shape[1]}, "
            f"range_min={float(temporal_range.min()):.6g}, "
            f"range_p01={float(np.percentile(temporal_range, 1)):.6g}, "
            f"range_p02={float(np.percentile(temporal_range, 2)):.6g}, "
            f"range_p05={float(np.percentile(temporal_range, 5)):.6g}, "
            f"range_max={float(temporal_range.max()):.6g}, "
            f"mean_max={float(temporal_mean.max()):.6g}, "
            f"count_mean_ge_value_min={int((temporal_mean >= float(args.lidar_value_min)).sum())}"
        )

    # Main rule: a saturated blocked sector remains almost constant at value 1.0
    # throughout the scan.
    strict_eps = float(args.lidar_stability_eps)
    value_min = float(args.lidar_value_min)
    stable_blocked = (temporal_range <= strict_eps) & (temporal_mean >= value_min)
    run = _circular_largest_true_run(stable_blocked, int(args.lidar_min_width))

    if run is None:
        # Some Webots/LiDAR pipelines introduce tiny numerical changes even inside a
        # forced constant sector. Relax only the temporal threshold, but still require
        # a high blocked/saturated value.
        relaxed_eps = max(strict_eps, float(np.percentile(temporal_range, float(args.lidar_relaxed_percentile))))
        relaxed_blocked = (temporal_range <= relaxed_eps) & (temporal_mean >= value_min)
        run = _circular_largest_true_run(relaxed_blocked, int(args.lidar_min_width))
        if run is not None:
            print(f"[MAPE-K] LiDAR localization used relaxed eps={relaxed_eps:.6g}. {_summary('stats')}")

    if run is None:
        # Last diagnostic fallback: if there is no stable blocked sector, this is
        # usually not a localization failure under a real persistent sector, but a
        # classifier false positive or an intermittent/dynamic anomaly. Do not invent
        # a simulator descriptor unless the caller explicitly asks to crash.
        debug = _summary("LiDAR localization failed")
        raise RuntimeError(debug)

    start, width = run
    values = []
    n = stack.shape[1]
    for i in range(width):
        values.append(stack[:, (start + i) % n])
    value = float(np.median(np.concatenate(values))) if values else 1.0
    return {
        "start": int(start),
        "width": int(width),
        "value": round(value, 6),
        "value_label": str(round(value, 6)),
        "n_scan_frames": len(scans),
        "temporal_range_max_in_region": round(float(max(temporal_range[(start + np.arange(width)) % n])), 8),
    }
def scan_and_localize(raw_env, modalities: set[str], settings):
    """Rotate the live robot once, capture both modalities, and localize requested faults.

    Active motion is executed with the existing Webots Supervisor, not with a
    second Robot instance. Returns (descriptors, errors) for trace logging.
    """
    if not modalities <= {"image", "lidar"} or not modalities:
        raise ValueError(f"Unsupported modalities: {modalities}")
    images, lidars = [], []
    raw_env.apply_action(5)
    raw_env.supervisor.step(raw_env.timestep)
    for degree in range(settings.scan_degrees):
        raw_env.apply_action(2)  # 1 degree per control action in this Webots environment
        raw_env.supervisor.step(raw_env.timestep)
        raw_env.get_reward()
        if degree % settings.scan_capture_every:
            continue
        obs = raw_env.get_obs()
        if "image" in modalities:
            images.append(np.asarray(obs["image"]).copy())
        if "lidar" in modalities:
            lidars.append(_latest_unpadded_lidar(obs["lidar"], raw_env.padding))
    out, errors = {}, {}
    for name in sorted(modalities):
        try:
            if name == "image":
                region = localize_image_region(images, settings)
                support = region.get("stable_area", region["width"] * region["height"])
                region["support_fraction"] = float(support) / float(np.prod(raw_env.img_size[:2]))
                region["full"] = region["support_fraction"] >= settings.full_region_fraction
            else:
                region = localize_lidar_region(lidars, settings)
                region["support_fraction"] = float(region["width"]) / float(raw_env.lidar.getHorizontalResolution())
                region["full"] = region["support_fraction"] >= settings.full_region_fraction
            out[name] = region
        except (RuntimeError, ValueError) as exc:
            errors[name] = str(exc)
    return out, errors
