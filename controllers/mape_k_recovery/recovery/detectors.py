"""Original paper detector architectures, checkpoint-compatible (state_dict keys retained)."""

import torch
from torch import nn
from torch.nn import functional as F

class ConvGNAct(nn.Module):

    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        g = 8 if c_out >= 32 else 4
        self.block = nn.Sequential(nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False), nn.GroupNorm(g, c_out), nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class ResidualBlock(nn.Module):

    def __init__(self, c):
        super().__init__()
        self.conv1 = ConvGNAct(c, c)
        self.conv2 = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.GroupNorm(8 if c >= 32 else 4, c))
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.conv2(self.conv1(x)))

class PatchMILImageClassifier(nn.Module):

    def __init__(self, topk_fraction=0.03):
        super().__init__()
        self.topk_fraction = float(topk_fraction)
        self.backbone = nn.Sequential(ConvGNAct(5, 32, 5, 2, 2), ConvGNAct(32, 64, 3, 2, 1), ResidualBlock(64), ConvGNAct(64, 96, 3, 2, 1), ResidualBlock(96), ConvGNAct(96, 128, 3, 1, 1))
        self.score_head = nn.Sequential(ConvGNAct(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))

    @staticmethod
    def _local_features(x):
        gray = x.mean(1, keepdim=True)
        mean = F.avg_pool2d(gray, 9, 1, 4)
        mean2 = F.avg_pool2d(gray * gray, 9, 1, 4)
        std = torch.sqrt(torch.clamp(mean2 - mean * mean, min=0) + 1e-06)
        dx = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        dy = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        return torch.cat([x, std, 0.5 * (dx + dy)], 1)

    def forward(self, x):
        sm = self.score_head(self.backbone(self._local_features(x)))
        flat = sm.flatten(1)
        k = max(1, int(round(flat.shape[1] * self.topk_fraction)))
        return torch.topk(flat, k=k, dim=1).values.mean(1)

class PersistenceLiDARClassifier(nn.Module):
    TH = (0.95, 0.96, 0.98, 0.99)

    def __init__(self, dropout=0.1):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(12, 32, 9, 2, 4), nn.ReLU(), nn.Conv1d(32, 64, 9, 2, 4), nn.ReLU(), nn.Conv1d(64, 64, 7, 2, 3), nn.ReLU())
        self.ap = nn.AdaptiveAvgPool1d(16)
        self.mp = nn.AdaptiveMaxPool1d(16)
        self.head = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    @classmethod
    def feats(cls, x):
        mean = x.mean(1)
        std = x.std(1, unbiased=False)
        mn = x.min(1).values
        mx = x.max(1).values
        rng = mx - mn
        frac = [(x >= t).float().mean(1) for t in cls.TH]
        stable = [((mn >= t) & (rng <= 0.015)).float() for t in (0.95, 0.96, 0.98)]
        return torch.stack([mean, std, mn, mx, rng] + frac + stable, 1)

    def forward(self, x):
        h = self.spatial(self.feats(x))
        return self.head(torch.cat([self.ap(h), self.mp(h)], 1).flatten(1)).squeeze(1)
