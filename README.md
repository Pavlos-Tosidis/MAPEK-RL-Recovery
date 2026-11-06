# MAPE-K Recovery for Multimodal Robotic Navigation Under Sensor Anomalies

**Webots controller implementation** of the targeted recovery approach described in

> Pavlos Tosidis, Nikolaos Passalis, and Anastasios Tefas, **“MAPE-K Recovery for Multimodal Robotic Navigation Under Sensor Anomalies,”** *IEEE Access*, vol. 14, pp. 160172–160190, 2026. DOI: [10.1109/ACCESS.2026.3743649](https://doi.org/10.1109/ACCESS.2026.3743649).

A deployed **multimodal PPO policy** combines RGB, LiDAR and localization-derived task state.
When learned sensor detectors confirm a persistent abnormality, the controller temporarily stops navigation,
rotates to localize corruption fixed in sensor coordinates, reproduces the diagnosed corruption in its Webots
environment, and finetunes the **original nominal PPO checkpoint**. The adapted policy is saved and redeployed.

This repository focuses on **using, training and extending the method** rather than re-running every comparison from the article.

## Capabilities

| Operation | Description |
| --- | --- |
| Sensor data collection | Nominal or persistent-fault RGB / 8-frame LiDAR / task-state sequences from Webots; rotating controller or optional PPO policy |
| Anomaly classifier training | 503,137-parameter RGB Patch-MIL and 591,777-parameter LiDAR persistence Conv1D; partial/full faults, cross-modal negatives, sequence-group splitting, threshold calibration |
| Nominal PPO training | Full `image + lidar + world_info` input, default 15,000,000 steps |
| Standalone inference | Load nominal or recovered PPO checkpoints under normal or selected persistent faults |
| Active fault localization | 360° rotation, image stable-pixel analysis, LiDAR stable blocked-sector analysis |
| Targeted recovery | PPO finetuning from the **nominal** checkpoint under the *estimated* corruption, default 200,000 steps |
| MAPE-K runtime | Learned-detector confirmation, localization, finetune, checkpoint save and redeployment |
| Detector re-baselining | Optional separate data capture and fine-tuning initialized from an existing detector checkpoint |

Anomaly detectors assess **sensor condition**; they are not the encoders inside the PPO policy.

## Project layout

```text
webots-project/
├── worlds/                   # add original .wbt worlds
├── protos/                   # add original .proto definitions
├── textures/                 # add original texture assets
├── controllers/
│   └── mape_k_recovery/
│       ├── mape_k_recovery.py     # all CLI entry points
│       └── recovery/
│           ├── webots_env.py      # scene, observations, actions, rewards
│           ├── policy_models.py   # original PPO feature-extraction network
│           ├── dataset.py         # observation collection + dataset splits
│           ├── corruptions.py     # image masks and LiDAR close-return faults
│           ├── train_detector.py  # training, validation, calibration
│           ├── train_policy.py    # full-input PPO training
│           ├── monitor.py         # learned anomaly inference
│           ├── localization.py    # 360° scan, image and LiDAR localization
│           ├── adaptation.py      # targeted PPO recovery
│           ├── mapek.py           # health confirmation and Knowledge state
│           └── infer.py           # standalone and adaptive execution
├── configs/recovery_descriptor.example.json
├── requirements.txt
├── CITATION.cff
└── README.md
```

**Webots assets:** The uploaded source contained Python code but no `.wbt`, `.proto` or texture files.
Copy the existing assets from the matching original Webots project into the folders above.
`webots_env.py` deliberately keeps the original world object DEF names, robot device names, navigation reward and discrete actions.
It is not a generic Webots TurtleBot wrapper.

## Install

Requirements: a Webots installation capable of running Python robot controllers, Python 3.10+ compatible with your Webots version, and a CPU/CUDA PyTorch build supported by your machine.
The packages used for PPO are Stable-Baselines3 and Gymnasium; Ultralytics is required by the original task's person-detection action.

```bash
cd /path/to/webots-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate CPU/CUDA PyTorch build for your system if needed.
python -m pip install -r requirements.txt
```

The Python `controller` module comes from **Webots**, not pip.
The original environment loads `yolo11n.pt` for the person-detection action; allow the model to download once or provide it locally.

## Connect the Webots controller

Set your robot's Webots fields to use an external controller with supervisor privileges:

```webots
supervisor TRUE
controller "<extern>"
```

Start the world in Webots. From another terminal, with the virtual environment active:

```bash
export WEBOTS_HOME=/path/to/webots
RUN="$WEBOTS_HOME/webots-controller controllers/mape_k_recovery/mape_k_recovery.py"
$RUN --help
```

Use a **new external controller session** when changing between collection, training and inference.
Set the Webots Python command to the virtual-environment interpreter.
You can alternatively choose `controller "mape_k_recovery"` and populate Webots `controllerArgs`.
All paths below are relative to the project root; absolute paths are also accepted.

## 1. Collect Webots observations

No pretrained agent is necessary. The collector rotates the robot, periodically repositions it and captures
separate episode sequences. For more varied trajectories, provide a nominal trained PPO checkpoint.

```bash
$RUN collect \
  --output datasets/nominal_v1 \
  --episodes 220 \
  --steps-per-episode 150 \
  --stride 5 \
  --seed 42
```

**Optional PPO-guided collection:**

```bash
$RUN collect \
  --output datasets/nominal_ppo_v1 \
  --checkpoint outputs/policies/nominal_v1/model.zip \
  --episodes 220 \
  --steps-per-episode 150 \
  --stride 5
```

The output includes `manifest.json` and `sequences/episode_*.npz`. Sequence IDs form
train/validation groups: **frames from the same episode never appear in both partitions**.
The training dataset applies controlled synthetic faults: localized RGB rectangles, ellipses, polygons,
blobs and strips in black/gray/random colors at 5/10/20/35% severity, full-frame failure, LiDAR
close-range sectors with multiple ranges/severities, and full-scan failure. When training one detector,
a fault in the *other* modality supplies a cross-modal negative.

## 2. Train the two learned anomaly detectors

Detector training runs **outside Webots**; use normal Python and the same virtual environment:

```bash
python controllers/mape_k_recovery/mape_k_recovery.py train-detector \
  --modality image \
  --data datasets/nominal_v1 \
  --output outputs/detectors/image_v1 \
  --epochs 20 --batch-size 4 --device auto

python controllers/mape_k_recovery/mape_k_recovery.py train-detector \
  --modality lidar \
  --data datasets/nominal_v1 \
  --output outputs/detectors/lidar_v1 \
  --epochs 20 --batch-size 32 --device auto
```

Each run selects the best validation model and calibrates a raw-logit threshold from validation predictions.
Outputs include the detector checkpoint and `metadata.json` with parameter count, data provenance,
validation groups, augmentation settings and calibrated threshold.

**Inspect detector inference within Webots:**

```bash
$RUN infer-detector \
  --image-detector outputs/detectors/image_v1/image_detector.pt \
  --lidar-detector outputs/detectors/lidar_v1/lidar_detector.pt \
  --condition partial_lidar --steps 100
```

## 3. Train the nominal multimodal PPO policy

The nominal training configuration uses `MultiInputPolicy` with the source image, LiDAR and task-state
feature extractors. The default is **15 million** steps; PPO settings are shared with the research
implementation: 2,048 rollout steps, batch size 128, gamma 0.99, GAE lambda 0.95, 10 epochs,
entropy coefficient 0.01, value coefficient 0.5 and target KL 0.03.

```bash
$RUN train-policy \
  --output outputs/policies/nominal_v1 \
  --steps 15000000 \
  --save-freq 2000000 \
  --device auto
```

Outputs: `model.zip`, `vecnormalize.pkl`, `metadata.json` and training logs.

**Task interface**

| Input / action | Representation |
| --- | --- |
| RGB | 270 × 480 × 3 |
| LiDAR | 8 × 1200 circularly padded, normalized proximity |
| World/task state | 3 values |
| Actions 0–1 | Forward/backward by 0.01 m |
| Actions 2–3 | Turn left/right by 1° |
| Action 4 | Person-detection / stop action |
| Action 5 | No motion |

Episode horizon is 500 actions; the source Webots environment contains the navigation reward,
collision and success logic.

## 4. Run independent PPO inference

```bash
$RUN infer-policy \
  --base-policy outputs/policies/nominal_v1/model.zip \
  --condition normal --episodes 5 \
  --output outputs/inference/nominal_v1
```

To apply persistent synthetic sensor faults during deployment, choose
`partial_image`, `full_image`, `partial_lidar`, `full_lidar`, `partial_both` or `full_both`.
Policy inference samples actions **stochastically by default**; pass `--deterministic` to use the PPO mean/mode.
The same `infer-policy` command can be used with the checkpoint produced by a recovery run.

## 5. Run MAPE-K fault detection, localization and recovery

```bash
$RUN infer-mapek \
  --base-policy outputs/policies/nominal_v1/model.zip \
  --image-detector outputs/detectors/image_v1/image_detector.pt \
  --lidar-detector outputs/detectors/lidar_v1/lidar_detector.pt \
  --condition partial_image \
  --episodes 1 --max-steps 500 \
  --onset 5 --scan-degrees 360 --scan-capture-every 5 \
  --finetune-steps 200000 --max-recoveries 1 \
  --output outputs/recovery/partial_image_v1
```

At each navigation step the controller scores camera and LiDAR with the learned models.
A sensor must produce five consecutive positive decisions before diagnosis begins.
After halting navigation (action 5), the robot rotates and collects observations approximately every 5°,
then identifies sensor-fixed image regions and/or LiDAR sectors. **Ground-truth fault coordinates are not
passed to localization.** The estimated fault descriptors configure the recovery simulator.
The original nominal checkpoint is loaded again and finetuned under that diagnosed condition with a fresh
reward-normalization estimator, using the **same Webots Supervisor process**. The new policy is saved and redeployed.
Results include `events.jsonl`, `knowledge.json`, and a `recovery_01/` directory with the adapted checkpoint.

Change `--condition` to `partial_lidar`, `full_image`, `full_lidar`, `partial_both` or `full_both`
to exercise other supported anomalies. Localization can reject an anomaly if stable support is not identified;
in that case the controller records the rejection and does not invent a fault descriptor.

## 6. Recover directly from a previously localized descriptor

For externally diagnosed faults, place a JSON file containing an `image` and/or `lidar` descriptor.
See `configs/recovery_descriptor.example.json`. Example:

```json
{
  "lidar": {"start": 240, "width": 108, "value": 0.992, "full": false}
}
```

Then run:

```bash
$RUN recover-policy \
  --base-policy outputs/policies/nominal_v1/model.zip \
  --descriptor configs/recovery_descriptor.example.json \
  --finetune-steps 200000 \
  --output outputs/recovery/from_descriptor_v1
```

Both `recover-policy` and the MAPE-K recovery always initialize PPO from the specified
**nominal** checkpoint, rather than chaining weights from a previous recovery.

## 7. Update a detector to a new reference condition (optional)

Policy recovery does not repair the physical sensor. To change a detector's *reference* condition,
collect sequences while the diagnosed fault is present and fine-tune the relevant classifier
from its previous checkpoint.

```bash
$RUN collect \
  --condition partial_image \
  --output datasets/image_fault_reference_v1 \
  --checkpoint outputs/policies/nominal_v1/model.zip \
  --episodes 220 --steps-per-episode 150 --stride 5

python controllers/mape_k_recovery/mape_k_recovery.py train-detector \
  --modality image \
  --data datasets/image_fault_reference_v1 \
  --initial-checkpoint outputs/detectors/image_v1/image_detector.pt \
  --output outputs/detectors/image_rebased_v1 \
  --epochs 20 --batch-size 4
```

This is a **separate detector-update workflow**. The runtime does not automatically substitute updated
detector checkpoints after recovery. In particular, repeated multi-cycle detection under an updated
operating condition requires selecting/reloading the appropriate detector weights.

## Artifacts, configuration and limitations

Datasets and checkpoints are deliberately excluded from Git. All training runs record configuration and
dependency versions; output directories cannot silently overwrite an earlier run.
The method targets **persistent sensor-fixed corruptions** and assumes the Webots simulation accurately
represents the diagnosed sensing condition. Localization may be unreliable for intermittent faults,
moving occlusions or real-world scene regions that remain stable across a scan.
The automated runtime currently implements **policy** recovery and redeployment; classifier re-baselining
is an explicit separate operation. See `docs/IMPLEMENTATION.md` for the source versions and implementation scope.

## Attribution and license

The associated research was supported by the **RoboSAPIENS** project, funded by the European Union's
Horizon Europe research and innovation programme under grant agreement **No. 101133807**.

The software is distributed under the MIT License; preserve attributions and check license compatibility
with any assets or third-party code added to this repository. For scholarly use, cite the IEEE Access
article above and the repository's `CITATION.cff` software citation.

### BibTeX

```bibtex
@article{tosidis2026mapekrecovery,
  author  = {Pavlos Tosidis and Nikolaos Passalis and Anastasios Tefas},
  title   = {MAPE-K Recovery for Multimodal Robotic Navigation Under Sensor Anomalies},
  journal = {IEEE Access},
  volume  = {14},
  pages   = {160172--160190},
  year    = {2026},
  doi     = {10.1109/ACCESS.2026.3743649}
}
```
