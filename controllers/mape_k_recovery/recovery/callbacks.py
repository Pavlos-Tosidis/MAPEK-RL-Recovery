"""Episode-outcome TensorBoard summaries for training."""
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class TerminationCallback(BaseCallback):
    def __init__(self, window_size=100):
        super().__init__(verbose=0)
        self.recent = deque(maxlen=window_size)

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", ()), self.locals.get("infos", ())):
            if done:
                self.recent.append(dict(
                    success=int(bool(info.get("success", False))),
                    collision=int(bool(info.get("collided", False))),
                    timeout=int(bool(info.get("TimeLimit.truncated", False))),
                ))
        if len(self.recent) == self.recent.maxlen:
            for field in ("success", "collision", "timeout"):
                self.logger.record(f"rollout/{field}_last_{self.recent.maxlen}",
                                   sum(x[field] for x in self.recent))
        return True
