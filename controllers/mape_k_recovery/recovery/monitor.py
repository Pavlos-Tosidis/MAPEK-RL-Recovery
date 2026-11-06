"""Checkpointed, model-based anomaly evidence. No ground-truth fault labels."""
from pathlib import Path
from typing import Any
import numpy as np
import torch
from .detectors import PatchMILImageClassifier, PersistenceLiDARClassifier

MODELS = {"image": PatchMILImageClassifier, "lidar": PersistenceLiDARClassifier}


def normalize_image(image):
    x = np.asarray(image)
    if x.shape != (270, 480, 3):
        raise ValueError(f"Image must be 270x480x3, got {x.shape}")
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float() / 255.0


def normalize_lidar(lidar):
    x = np.asarray(lidar)
    if x.shape != (8, 1200):
        raise ValueError(f"LiDAR must be 8x1200, got {x.shape}")
    return torch.as_tensor(x.copy(), dtype=torch.float32)


class LearnedMonitor:
    def __init__(self, image_checkpoint: str | Path, lidar_checkpoint: str | Path,
                 device: str = "auto"):
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.models: dict[str, Any] = {}
        self.thresholds: dict[str, float] = {}
        self.metadata: dict[str, dict] = {}
        for modality, path in (("image", image_checkpoint), ("lidar", lidar_checkpoint)):
            state = torch.load(path, map_location=self.device, weights_only=False)
            if state.get("modality", modality) != modality:
                raise ValueError(f"Detector checkpoint has wrong modality: {path}")
            extra = {"topk_fraction": state.get("topk_fraction", .03)} if modality == "image" else {}
            model = MODELS[modality](**extra).to(self.device)
            model.load_state_dict(state["model_state_dict"], strict=True)
            model.eval()
            self.models[modality] = model
            self.thresholds[modality] = float(state["logit_threshold"])
            self.metadata[modality] = state.get("metadata", {})

    @torch.inference_mode()
    def analyze(self, observation: dict):
        inputs = {"image": normalize_image(observation["image"]),
                  "lidar": normalize_lidar(observation["lidar"])}
        evidence = {}
        for modality in ("image", "lidar"):
            score = float(self.models[modality](inputs[modality].unsqueeze(0).to(self.device)).item())
            evidence[f"{modality}_score"] = score
            evidence[f"{modality}_raw"] = score >= self.thresholds[modality]
        return evidence
