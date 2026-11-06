# Implementation notes

The clean controller is based on the supplied rapid-recovery Python archive.

- **Environment:** `turtlebot_environment_mapek_v11_realistic_faults.py` supplies the
  Webots task, actions, reward function, multimodal preprocessing, and persistent
  sensor-coordinate anomaly injection. Asset-specific DEF names are retained.
- **Policy extractor:** the encoder layout and names originate from
  `models_custom.py`, retaining the original 270×480 RGB and 8×1200 LiDAR inputs.
- **Detector architectures:** the original Patch-MIL and persistence-Conv1D shapes
  and parameter counts are preserved.
- **Active localization:** image stable-pixel components and circular LiDAR
  stable close-return sectors follow the supplied v18 recovery controller.
- **Recovery:** PPO always initializes from the supplied nominal checkpoint,
  recovers in the same Webots controller process, and uses fresh reward
  normalization statistics while finetuning.

The large source archive includes many historical experiment runners and metrics.
They are intentionally not copied into the reusable implementation. This
repository is not a full copy of the research evaluation scripts.

## Known scope boundaries

- Fault collection uses sensor-fixed synthetic corruption as implemented by the
  provided Webots environment, not physical fault injection.
- The dataset builder uses clean or reference-condition sequences with synthetic
  positive examples and modality-specific cross-negative examples. Source
  trajectories recorded under injected faults may be introduced as an additional
  dataset option.
- The runtime can redeploy an adapted PPO checkpoint, but it does not
  **automatically retrain and replace the anomaly-detector checkpoints** after
  recovery. Re-baselining is provided separately through `collect --condition`
  and `train-detector --initial-checkpoint`. Use that path to update detector
  reference conditions before claiming multi-cycle recovery.
- Mixed-modality partial/full combined faults are represented by the combined
  partial fault mode with estimated full-width/full-frame support.
- Webots world/proto/texture resources were not present in the uploaded ZIP.

## Checkpoint compatibility

The `models_custom.py` alias supports original checkpoint import paths where
possible. Checkpoint compatibility across Stable-Baselines3, PyTorch,
Gymnasium and Python releases is not guaranteed; matching network and
normalization settings is essential.
