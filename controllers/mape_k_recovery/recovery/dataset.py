"""Webots collection and sequence-separated synthetic anomaly datasets.

Collect only *nominal* sensor streams; augment the same clean sequences offline.
Split scene/episode IDs BEFORE augmenting so correlated image frames and LiDAR
windows never appear in both training and validation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import metadata, read_json, write_json, unique_directory
from .corruptions import apply_condition
from .monitor import normalize_image, normalize_lidar


CONDITIONS = ("normal", "cross_modal_negative", "partial", "full")


def collect(args):
    from stable_baselines3 import PPO
    from .runtime import make_environment, set_condition
    from . import policy_models  # SB3 checkpoint custom extractor

    output = unique_directory(args.output)
    (output / "sequences").mkdir()
    env = make_environment(seed=args.seed)
    policy = PPO.load(args.checkpoint, device=args.device) if args.checkpoint else None
    rng = np.random.default_rng(args.seed)
    sequences = []
    try:
        for episode in range(args.episodes):
            set_condition(env, getattr(args, "condition", "normal"), rng)
            images, lidars, worlds, timesteps = [], [], [], []
            obs, _ = env.reset()
            for step in range(args.steps_per_episode):
                if step % args.stride == 0:
                    images.append(np.asarray(obs["image"], dtype=np.uint8).copy())
                    lidars.append(np.asarray(obs["lidar"], dtype=np.float32).copy())
                    worlds.append(np.asarray(obs["world_info"], dtype=np.float32).copy())
                    timesteps.append(step)
                if policy is None:
                    # Rotate in both directions for multiple views, with occasional short forward motion.
                    action = 2 if (step // 25) % 2 == 0 else 3
                    if step % 41 == 40:
                        action = 0
                else:
                    action, _ = policy.predict(obs,
                                               deterministic=False)
                    action = int(action)
                obs, _, done, truncated, _ = env.step(action)
                if done or truncated:
                    break
            if images:
                filename = f"sequences/episode_{episode:05d}.npz"
                np.savez_compressed(output / filename, image=np.stack(images),
                                    lidar=np.stack(lidars), world_info=np.stack(worlds),
                                    timestep=np.asarray(timesteps, dtype=np.int32))
                sequences.append({"sequence_id": f"episode_{episode:05d}",
                                  "scenario_id": f"episode_{episode:05d}",
                                  "file": filename, "frames": len(images)})
            print(f"Captured sequence {episode+1}/{args.episodes}: {len(images)} frames")
    finally:
        env.supervisor.cleanup() if hasattr(env.supervisor, "cleanup") else None
    write_json(output / "manifest.json", metadata(
        "nominal_sensor_collection" if getattr(args, "condition", "normal") == "normal"
        else "reference_sensor_collection",
        seed=args.seed, episodes=args.episodes, condition=getattr(args, "condition", "normal"),
        stride=args.stride, steps_per_episode=args.steps_per_episode,
        control_method="ppo" if policy else "rotate_and_reposition",
        checkpoint=args.checkpoint, image_shape=[270,480,3], lidar_shape=[8,1200],
        lidar_normalization="proximity=1-(distance_m/12.5), capped at 12.5m, circular pad 60",
        camera_channels="original Webots camera first three BGRA channels",
        source="Webots supervisor", sequences=sequences))


def split_sequences(manifest: dict, fraction: float = .2, seed: int = 42):
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be between 0 and 1")
    groups = sorted(set(s["scenario_id"] for s in manifest["sequences"]))
    if len(groups) < 2:
        raise ValueError("At least two distinct scene/episode groups are required")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, min(len(groups)-1, round(fraction*len(groups))))
    val = set(groups[:n_val])
    train_rows = [s for s in manifest["sequences"] if s["scenario_id"] not in val]
    val_rows = [s for s in manifest["sequences"] if s["scenario_id"] in val]
    return train_rows, val_rows


@lru_cache(maxsize=4)
def _load_sequence(path: str):
    with np.load(path, allow_pickle=False) as file:
        return {key: file[key] for key in ("image", "lidar", "world_info")}


class SyntheticAnomalyDataset(Dataset):
    """Four balanced variants per nominal frame: nominal/cross/partial/full.

    Cross-negative for image => LiDAR fault but healthy image; and vice versa.
    A persistent LiDAR sector is applied coherently across its 8-frame window.
    """

    def __init__(self, source: str | Path, modality: str, sequences: list[dict],
                 seed: int, training: bool, max_frames: int | None = None):
        if modality not in ("image", "lidar"):
            raise ValueError(modality)
        self.modality = modality
        self.source = Path(source)
        self.seed = int(seed)
        self.training = training
        self.epoch = 0
        self.rows = []
        for entry in sequences:
            path = str((self.source / entry["file"]).resolve())
            for frame in range(int(entry["frames"])):
                self.rows.append((path, frame, entry["sequence_id"]))
        if max_frames is not None:
            rng = np.random.default_rng(seed)
            inds = rng.permutation(len(self.rows))[:max_frames]
            self.rows = [self.rows[int(i)] for i in sorted(inds)]
        if not self.rows:
            raise ValueError("No frames available for detector training")

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return len(self.rows) * len(CONDITIONS)

    def description(self, index: int):
        path, frame, seq_id = self.rows[index // 4]
        return {"sequence_id": seq_id, "frame_index": frame, "condition": CONDITIONS[index % 4],
                "modality": self.modality, "label": int(index % 4 >= 2)}

    def __getitem__(self, index: int):
        path, frame, _ = self.rows[index // 4]
        clean = _load_sequence(path)
        base = {key: data[frame] for key, data in clean.items()}
        variant = index % 4
        # Training augmentations vary across epochs; validation is fixed.
        epoch = self.epoch if self.training else 0
        seed = self.seed + int(index) * 32771 + epoch * 104729
        rng = np.random.default_rng(seed)
        other = "lidar" if self.modality == "image" else "image"
        condition = ("normal" if variant == 0 else
                     f"partial_{other}" if variant == 1 else
                     f"partial_{self.modality}" if variant == 2 else
                     f"full_{self.modality}")
        observed, _ = apply_condition(base, condition, rng)
        raw = observed[self.modality]
        x = normalize_image(raw) if self.modality == "image" else normalize_lidar(raw)
        return x, torch.tensor(float(variant >= 2), dtype=torch.float32)
