"""PPO feature-extraction network from the source project.

The fixed image CNN is deliberately large and accepts a 270x480 camera after
SB3's HWC->CHW conversion. Class names and parameter keys are retained for
compatibility with source checkpoints using CustomCombinedExtractor.
"""

import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(107520, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(self.cnn(x))


class LiDAREncoder(nn.Module):
    def __init__(self, input_length=1200, out_dim=128):
        super().__init__()
        if input_length != 1200:
            raise ValueError("Source LiDAR encoder expects 1200 samples")
        self.out_dim = 128  # The original code fixed this output dimension.
        self.conv1 = nn.Conv1d(8, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(32)  # Retained for checkpoint state_dict compatibility.
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(128)  # Unused in the original forward pass.
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc = nn.Sequential(
            nn.Linear(9600, 512), nn.ReLU(), nn.Linear(512, 128), nn.ReLU()
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.fc(self.flatten(x))


class WorldEncoder(nn.Module):
    def __init__(self, out_dim=3):
        super().__init__()
        self.out_dim = out_dim
        self.flat = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        return self.flat(x)


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim=1)
        if isinstance(observation_space, spaces.Box):
            self.o_space = "image_only"
            self.image_encoder = ImageEncoder(out_dim=512)
            self.extractor = self.image_encoder
            self.extractors = nn.ModuleDict()
            total = 512
        else:
            self.o_space = "full"
            extractors = {}
            total = 0
            for key in observation_space.spaces:
                if key == "image":
                    self.image_encoder = ImageEncoder(out_dim=512)
                    extractors[key] = self.image_encoder
                    total += 512
                elif key == "lidar":
                    self.lidar_encoder = LiDAREncoder()
                    extractors[key] = self.lidar_encoder
                    total += 128
                elif key == "world_info":
                    self.world_encoder = WorldEncoder()
                    extractors[key] = self.world_encoder
                    total += 3
                else:
                    raise ValueError(f"Unsupported observation: {key}")
            self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total

    def forward(self, observations):
        if self.o_space == "image_only":
            return self.extractor(observations)
        return torch.cat([encoder(observations[key]) for key, encoder in self.extractors.items()], dim=1)
