"""Direct learned-detector inference on Webots sensor observations."""
import numpy as np
from .runtime import make_environment, set_condition
from .monitor import LearnedMonitor


def run(args):
    rng = np.random.default_rng(args.seed)
    env = make_environment(seed=args.seed)
    monitor = LearnedMonitor(args.image_detector, args.lidar_detector, args.device)
    try:
        set_condition(env, args.condition, rng)
        obs, _ = env.reset()
        for t in range(args.steps):
            scores = monitor.analyze(obs)
            print(t, scores)
            obs, _, terminated, truncated, _ = env.step(2)
            if terminated or truncated:
                set_condition(env, args.condition, rng)
                obs, _ = env.reset()
    finally:
        env.supervisor.cleanup() if hasattr(env.supervisor, "cleanup") else None
