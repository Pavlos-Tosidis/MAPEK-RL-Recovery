"""Train a nominal full-observation PPO policy (15M steps by default)."""
from __future__ import annotations
import random
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from .callbacks import TerminationCallback
from .config import metadata, write_json, unique_directory
from .runtime import make_environment, make_vec_env
from .policy_models import CustomCombinedExtractor


def linear_schedule(value):
    return lambda progress_remaining: progress_remaining * value


def run(args):
    out = unique_directory(args.output)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    raw = make_environment(seed=args.seed)
    env = make_vec_env(raw, training=True, monitor_path=out / "monitor")
    params = dict(n_steps=args.n_steps, batch_size=args.batch_size, gamma=.99,
                  gae_lambda=.95, n_epochs=10, ent_coef=.01, vf_coef=.5,
                  target_kl=.03, learning_rate=linear_schedule(args.lr),
                  policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor,
                                     net_arch={"pi": [256,128], "vf": [256,128]},
                                     activation_fn=torch.nn.ReLU),
                  verbose=args.verbose, device=args.device, seed=args.seed,
                  tensorboard_log=str(out / "tensorboard"))
    write_json(out / "metadata.json", metadata("nominal_ppo", seed=args.seed,
               observations=["image","lidar","world_info"], steps=args.steps,
               ppo={"n_steps":args.n_steps,"batch_size":args.batch_size,"gamma":.99,
                    "gae_lambda":.95,"n_epochs":10,"ent_coef":.01,"vf_coef":.5,
                    "target_kl":.03,"lr_initial_linear":args.lr,
                    "net_arch":{"pi":[256,128],"vf":[256,128]}},
               normalize_observations=False,normalize_rewards=True,
               actions={"0":"forward_0.01m","1":"backward_0.01m","2":"left_1degree",
                        "3":"right_1degree","4":"detect","5":"no_motion"}))
    try:
        model=PPO("MultiInputPolicy",env,**params)
        callbacks=[TerminationCallback()]
        if args.save_freq>0:
            callbacks.append(CheckpointCallback(save_freq=args.save_freq,
                             save_path=str(out / "checkpoints"),name_prefix="nominal_policy"))
        model.learn(total_timesteps=args.steps,callback=CallbackList(callbacks))
        model.save(str(out / "model.zip"))
        env.save(str(out / "vecnormalize.pkl"))
        print(f"Saved nominal PPO policy in {out}")
    finally:
        env.close()
