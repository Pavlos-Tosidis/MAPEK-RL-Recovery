import torch
from recovery.detectors import PatchMILImageClassifier,PersistenceLiDARClassifier

def test_parameter_counts():
    a,b=PatchMILImageClassifier(),PersistenceLiDARClassifier()
    assert sum(p.numel() for p in a.parameters())==503137
    assert sum(p.numel() for p in b.parameters())==591777

def test_lidar_forward():
    m=PersistenceLiDARClassifier().eval()
    with torch.no_grad():
        out=m(torch.rand(2,8,1200))
    assert tuple(out.shape)==(2,)
