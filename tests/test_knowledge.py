from recovery.mapek import RecoveryKnowledge

def test_confirmation_and_cooldown():
    k=RecoveryKnowledge('base.zip','base.zip',onset=5,cooldown_steps=3)
    evidence={'image_raw':True,'lidar_raw':False}
    for _ in range(4):assert not k.update(evidence)
    assert k.update(evidence)=={'image'}
    k.reject({'image'},{'image':'no stable component'})
    for _ in range(3):assert not k.update(evidence)
    assert not k.update(evidence)

def test_redeployment_state():
    k=RecoveryKnowledge('base.zip','base.zip')
    k.redeployed({'image':{'x':2}},'recovered/model.zip')
    assert k.adaptation_count==1
    assert k.current_state=='recovered'
    assert k.recovered_modalities=={'image'}
    for _ in range(6):assert not k.update({'image_raw':True,'lidar_raw':False})
