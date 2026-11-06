"""Train, calibrate, validate and export the original anomaly detector models."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import SyntheticAnomalyDataset, split_sequences
from .detectors import PatchMILImageClassifier, PersistenceLiDARClassifier
from .config import read_json, write_json, metadata, unique_directory


def threshold_from_validation(scores: np.ndarray, labels: np.ndarray):
    """Maximize balanced accuracy. Threshold is midpoint between distinct scores."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(scores) != len(labels) or not len(scores):
        raise ValueError("Invalid scores and labels")
    if set(labels) != {0, 1}:
        raise ValueError("Threshold calibration requires both classes")
    uniq = np.unique(scores)
    candidates = np.r_[uniq[0]-1, (uniq[:-1]+uniq[1:])/2, uniq[-1]+1]
    positives = labels == 1
    negatives = ~positives
    best_threshold = float(candidates[0])
    best = -1.0
    for cut in candidates:
        preds = scores >= cut
        balanced_accuracy = .5 * (preds[positives].mean() + (~preds[negatives]).mean())
        if balanced_accuracy > best:
            best, best_threshold = float(balanced_accuracy), float(cut)
    preds = scores >= best_threshold
    return best_threshold, dict(
        balanced_accuracy=best, tp=int((preds & positives).sum()),
        tn=int(((~preds) & negatives).sum()), fp=int((preds & negatives).sum()),
        fn=int(((~preds) & positives).sum()))


def validate(model, loader, device):
    model.eval()
    scores, targets, losses = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch, y in loader:
            z = model(batch.to(device))
            losses.append(float(loss_fn(z, y.to(device)).item()))
            scores.append(z.cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(scores), np.concatenate(targets), float(np.mean(losses))


def run(args):
    source = Path(args.data).expanduser().resolve()
    manifest = read_json(source / "manifest.json")
    if manifest.get("kind") not in ("nominal_sensor_collection", "reference_sensor_collection"):
        raise ValueError("Expected nominal Webots capture manifest")
    train_sequences, val_sequences = split_sequences(manifest, args.val_fraction, args.seed)
    out = unique_directory(args.output)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_ds = SyntheticAnomalyDataset(source, args.modality, train_sequences,
                                       seed=args.seed, training=True, max_frames=args.max_train_frames)
    val_ds = SyntheticAnomalyDataset(source, args.modality, val_sequences,
                                     seed=args.seed+123, training=False, max_frames=args.max_val_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    model = PatchMILImageClassifier() if args.modality == "image" else PersistenceLiDARClassifier()
    expected = 503137 if args.modality == "image" else 591777
    assert sum(p.numel() for p in model.parameters()) == expected
    if getattr(args, "initial_checkpoint", None):
        initial=torch.load(args.initial_checkpoint,map_location='cpu',weights_only=False)
        if initial.get('modality')!=args.modality:
            raise ValueError('Detector checkpoint modality does not match --modality')
        model.load_state_dict(initial['model_state_dict'],strict=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        running = []
        for x, target in train_loader:
            x, target = x.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), target)
            loss.backward()
            optimizer.step()
            running.append(float(loss.item()))
        scores, labels, val_loss = validate(model, val_loader, device)
        print(f"epoch={epoch+1}/{args.epochs} train_BCE={np.mean(running):.5f} val_BCE={val_loss:.5f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    scores, labels, val_loss = validate(model, val_loader, device)
    cut, metrics = threshold_from_validation(scores, labels)
    info = metadata("anomaly_detector", modality=args.modality, seed=args.seed,
                    architecture=type(model).__name__, parameter_count=expected,
                    train_sequences=[s["sequence_id"] for s in train_sequences],
                    validation_sequences=[s["sequence_id"] for s in val_sequences],
                    train_items=len(train_ds), validation_items=len(val_ds),
                    generated_conditions=list(("normal", "cross_modal_negative", "partial", "full")),
                    image_shapes=list(("rectangle", "ellipse", "polygon", "blob", "strip")),
                    image_contents=["black", "gray", "color"],
                    severity_fractions=[.05,.10,.20,.35],
                    lidar_distance_m=[.05,.10,.20,.30,.50],
                    validation_metrics=metrics, threshold_calibration="balanced_accuracy_on_validation",
                    epochs=args.epochs, learning_rate=args.lr,
                    original_capture_manifest=str(source / "manifest.json"),
                    reference_condition=manifest.get("condition","normal"),
                    initialization_checkpoint=getattr(args,"initial_checkpoint",None))
    checkpoint = {"model_state_dict": best_state, "modality": args.modality,
                  "logit_threshold": cut, "metadata": info}
    if args.modality == "image":
        checkpoint["topk_fraction"] = .03
    torch.save(checkpoint, out / f"{args.modality}_detector.pt")
    write_json(out / "metadata.json", info)
    print(f"Saved detector: {out}, threshold={cut:.6f}, validation={metrics}")
