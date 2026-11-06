"""Single Webots Supervisor instance, vector environment and scenario control."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .corruptions import image_mask, corrupt_lidar


def make_environment(seed=3, **kwargs):
    from .webots_env import Env  # Webots controller module is only needed at runtime
    env = Env(seed=seed, action_space="discrete",
              observation_space=["image", "lidar", "world_info"],
              inference_mode=kwargs.get("inference_mode", False))
    env.anomalies_enabled = False
    env.clear_experiment_anomaly()
    return env


def make_vec_env(raw_env, training=False, monitor_path=None):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    wrapped = Monitor(raw_env, filename=str(monitor_path) if monitor_path else None)
    vec = DummyVecEnv([lambda: wrapped])
    return VecNormalize(vec, norm_obs=False, norm_reward=training, clip_reward=10.0, training=training)


def set_condition(env, condition: str, rng: np.random.Generator):
    """Install persistent known conditions for demonstration and data generation.

    Actual recovery/localization does not inspect these privileged descriptors.
    """
    if condition == "normal":
        env.clear_experiment_anomaly()
        return
    image_support = image_info = lidar_info = None
    if condition.endswith(("image", "both")) and condition.startswith("partial"):
        image_support, image_info = image_mask(270, 480, rng)
        variant = str(rng.choice(("black", "gray", "random")))
        image_info["color"] = ([0,0,0] if variant == "black" else [127,127,127] if variant == "gray"
                               else [int(x) for x in rng.integers(0,256,size=3)])
    if condition.endswith(("lidar", "both")) and condition.startswith("partial"):
        _, lidar_info = corrupt_lidar(np.zeros((8,1200),dtype=np.float32),rng)
    env.set_experiment_anomaly(condition, image_mask=image_support, image_spec=image_info, lidar_spec=lidar_info)


def squeeze_obs(observation):
    return {key: np.asarray(value)[0] for key,value in observation.items()}


def resolve_model(checkpoint):
    from stable_baselines3 import PPO
    from . import policy_models
    return PPO.load(str(checkpoint), device="auto")
