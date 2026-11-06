"""Compatibility aliases for PPO checkpoints saved by the original project.

Historic Stable-Baselines3 checkpoints may reference `models_custom.*`.
The actual implementation is maintained in `recovery.policy_models`.
"""
from recovery.policy_models import ImageEncoder, LiDAREncoder, WorldEncoder, CustomCombinedExtractor
