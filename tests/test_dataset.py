import numpy as np
from recovery.dataset import split_sequences
from recovery.corruptions import apply_condition

def test_split_by_scenario():
    m={'sequences':[{'sequence_id':str(i),'scenario_id':f's{i//2}'} for i in range(12)]}
    a,b=split_sequences(m,.25,123)
    assert {x['scenario_id'] for x in a}.isdisjoint({x['scenario_id'] for x in b})
    assert len(a)+len(b)==12

def test_cross_modal_fault():
    rng=np.random.default_rng(2)
    base={'image':np.ones((270,480,3),dtype=np.uint8)*110,
          'lidar':np.ones((8,1200),dtype=np.float32)*.4,
          'world_info':np.ones(3,dtype=np.float32)}
    x,_=apply_condition(base,'partial_image',rng)
    assert not np.array_equal(x['image'],base['image'])
    assert np.array_equal(x['lidar'],base['lidar'])

