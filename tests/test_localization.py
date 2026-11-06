import numpy as np
from types import SimpleNamespace
from recovery.localization import _circular_largest_true_run,localize_lidar_region,localize_image_region

def settings():
    return SimpleNamespace(image_stability_tol=2.,image_morph_kernel=3,
        image_max_component_fraction=.6,image_min_component_area=80,
        lidar_stability_eps=.01,lidar_value_min=.95,lidar_min_width=20,
        lidar_relaxed_percentile=2.)

def test_circular_sector():
    mask=np.zeros(1080,bool)
    mask[np.arange(1060,1100)%1080]=True
    start,width=_circular_largest_true_run(mask,20)
    assert width==40
    assert start%1080==1060

def test_localize_lidar_fixed_sector():
    rng=np.random.default_rng(3)
    scans=[rng.uniform(.1,.8,1080).astype(np.float32) for _ in range(8)]
    for scan in scans:scan[170:230]=.992
    result=localize_lidar_region(scans,settings())
    assert result['start']==170
    assert result['width']==60
    assert abs(result['value']-.992)<.001

def test_localize_image_fixed_patch():
    rng=np.random.default_rng(3)
    frames=[rng.integers(30,255,(90,120,3),dtype=np.uint8) for _ in range(8)]
    for frame in frames:frame[24:45,31:64]=0
    result=localize_image_region(frames,settings())
    assert result['x']==31 and result['y']==24
    assert result['width']==33 and result['height']==21
