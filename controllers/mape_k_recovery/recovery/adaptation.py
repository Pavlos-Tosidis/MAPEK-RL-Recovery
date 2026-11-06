"""Localized anomaly injection and targeted PPO finetuning in one Webots controller."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.running_mean_std import RunningMeanStd
from . import policy_models  # make the custom feature extractor available to SB3
from .config import metadata, write_json, unique_directory

# These fields are needed *only* to restore the simulated live sensing condition
# after a recovery run; they are never provided to the active localizer.
FAULT_FIELDS = ("ano_mode", "ano_x", "ano_y", "ano_width", "ano_height",
                "ano_col", "startingpoint", "leng", "forced_anomaly_mode",
                "forced_image_region", "forced_lidar_region",
                "forced_image_mask", "forced_image_spec", "forced_lidar_obstruction")

def snapshot_fault(env):
    return {name: copy.deepcopy(getattr(env, name)) for name in FAULT_FIELDS}

def restore_fault(env, state):
    for name, value in state.items():
        setattr(env, name, copy.deepcopy(value))
    if state["ano_mode"] == "normal":
        env.clear_experiment_anomaly()
    else:
        # Don't call clear_experiment_anomaly(): it would lose persistent masks.
        env._apply_forced_anomaly()

def inject_diagnosed_fault(raw_env, descriptors):
    """Configure recovery from *estimated* descriptors, not simulator truth."""
    if not descriptors or not set(descriptors) <= {"image", "lidar"}:
        raise ValueError("Need at least one recognized localized fault descriptor")
    image = descriptors.get("image")
    lidar = descriptors.get("lidar")
    if image is not None and lidar is not None:
        # The environment supports combined faults through the experiment API.
        # Use estimated image bounding box and estimated LiDAR sector.
        mask=np.zeros(raw_env.img_size[:2], dtype=bool)
        if image.get("full", False):
            mask[:]=True
        else:
            x,y=int(image["x"]),int(image["y"])
            w,h=int(image["width"]),int(image["height"])
            mask[max(0,y):min(mask.shape[0],y+h),max(0,x):min(mask.shape[1],x+w)]=True
        image_spec={"color":list(image.get("color",(0,0,0))),
                    "bounding_rect":[int(image["x"]),int(image["y"]),int(image["width"]),int(image["height"])]}
        lidar_spec={"start":int(lidar["start"]),"width":int(lidar["width"]),
                    "value":float(lidar.get("value",1.0)),"noise_std":0.0}
        raw_env.set_experiment_anomaly("partial_both",image_mask=mask,
                                      image_spec=image_spec,lidar_spec=lidar_spec)
    elif image is not None:
        raw_env.set_detected_anomaly("image", image,
                    "full_image" if image.get("full",False) else "partial_image")
    else:
        raw_env.set_detected_anomaly("lidar", lidar,
                    "full_lidar" if lidar.get("full",False) else "partial_lidar")

def finetune(raw_env, vec_env, *, base_policy: str, descriptors: dict,
             steps: int, output: str, device="auto", seed=3):
    """Return an adapted PPO model, always starting from the original checkpoint.

    The existing Webots environment is reused; no second Supervisor is created.
    Reward normalization has fresh statistics for each adaptation.
    """
    if steps <= 0:
        raise ValueError("finetune steps must be positive")
    out=unique_directory(output)
    original=snapshot_fault(raw_env)
    old_training,old_reward=vec_env.training,vec_env.norm_reward
    try:
        inject_diagnosed_fault(raw_env,descriptors)
        vec_env.ret_rms=RunningMeanStd(shape=())
        vec_env.returns=np.zeros(vec_env.num_envs,dtype=np.float64)
        vec_env.training=True
        vec_env.norm_reward=True
        model=PPO.load(str(base_policy),env=vec_env,device=device)
        model.set_random_seed(seed)
        model.learn(total_timesteps=steps,reset_num_timesteps=False)
        model.save(str(out/'model.zip'))
        vec_env.save(str(out/'vecnormalize.pkl'))
        write_json(out/'metadata.json',metadata('targeted_policy_recovery',
                   base_checkpoint=str(Path(base_policy).expanduser().resolve()),
                   descriptors=descriptors,requested_steps=steps,seed=seed,
                   model_path='model.zip',normalize_observations=False,
                   normalize_rewards_during_recovery=True))
        return model,out
    finally:
        restore_fault(raw_env,original)
        vec_env.training=old_training
        vec_env.norm_reward=old_reward
