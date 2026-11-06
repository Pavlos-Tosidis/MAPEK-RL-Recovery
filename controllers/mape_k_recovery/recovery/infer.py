"""Independent inference and complete MAPE-K recovery in Webots."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from stable_baselines3 import PPO
from . import policy_models  # SB3 deserialization
from .runtime import make_environment, make_vec_env, set_condition, squeeze_obs
from .monitor import LearnedMonitor
from .localization import scan_and_localize
from .mapek import RecoveryKnowledge
from .adaptation import finetune
from .config import unique_directory,write_json

def _log(out,records,name='events.jsonl'):
    path=Path(out)/name
    with path.open('w',encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record,default=str)+'\n')

def infer_policy(args):
    out=unique_directory(args.output)
    raw=make_environment(seed=args.seed,inference_mode=True)
    vec=make_vec_env(raw,training=False,monitor_path=out/'monitor')
    model=PPO.load(args.base_policy,env=vec,device=args.device)
    rng=np.random.default_rng(args.seed)
    rows=[]
    try:
        for episode in range(args.episodes):
            set_condition(raw,args.condition,rng)
            obs=vec.reset()
            for step in range(args.max_steps):
                action,_=model.predict(obs,deterministic=args.deterministic)
                obs,reward,done,infos=vec.step(action)
                rows.append({'episode':episode,'step':step,'action':int(action[0]),
                             'reward':float(reward[0]),'condition':args.condition})
                if bool(done[0]):break
        _log(out,rows)
        print(f'Policy inference completed. Traces in {out}')
    finally:
        vec.close()

def infer_mapek(args):
    out=unique_directory(args.output)
    raw=make_environment(seed=args.seed,inference_mode=True)
    vec=make_vec_env(raw,training=False,monitor_path=out/'monitor')
    detector=LearnedMonitor(args.image_detector,args.lidar_detector,args.device)
    model=PPO.load(args.base_policy,env=vec,device=args.device)
    knowledge=RecoveryKnowledge(base_model=args.base_policy,active_model=args.base_policy,
                               onset=args.onset,cooldown_steps=args.cooldown)
    rng=np.random.default_rng(args.seed)
    rows=[]
    try:
        for episode in range(args.episodes):
            set_condition(raw,args.condition,rng)
            obs=vec.reset()
            knowledge.reset_episode()
            for step in range(args.max_steps):
                evidence=detector.analyze(squeeze_obs(obs))
                confirmed=knowledge.update(evidence)
                if confirmed and knowledge.adaptation_count<args.max_recoveries:
                    # Halt normal navigation and collect sensor-fixed evidence.
                    raw.apply_action(5)
                    descriptors,errors=scan_and_localize(raw,confirmed,args)
                    rows.append({'episode':episode,'step':step,'event':'localization',
                                 'modalities':sorted(confirmed),'descriptors':descriptors,'errors':errors})
                    if descriptors and not errors:
                        recover_output=str(out/f'recovery_{knowledge.adaptation_count+1:02d}')
                        model,recover_dir=finetune(raw,vec,base_policy=args.base_policy,
                                                 descriptors=descriptors,
                                                 steps=args.finetune_steps,output=recover_output,
                                                 device=args.device,seed=args.seed+knowledge.adaptation_count)
                        knowledge.redeployed(descriptors,recover_dir/'model.zip')
                        print(f'[MAPE-K] Recovered {sorted(descriptors)} -> {recover_dir / "model.zip"}')
                    else:
                        knowledge.reject(confirmed,errors)
                    # Scanner and recovery advance the simulator outside VecEnv.step.
                    # Reset the rollout observation to prevent acting on a stale frame.
                    obs={key:np.expand_dims(np.asarray(value),0)
                         for key,value in raw.get_obs().items()}
                action,_=model.predict(obs,deterministic=args.deterministic)
                obs,reward,done,infos=vec.step(action)
                rows.append({'episode':episode,'step':step,'action':int(action[0]),
                             'reward':float(reward[0]),'condition':args.condition,
                             'state':knowledge.current_state,
                             'image_score':evidence['image_score'],
                             'lidar_score':evidence['lidar_score'],
                             'image_anomaly':evidence['image_raw'],
                             'lidar_anomaly':evidence['lidar_raw'],
                             'recoveries':knowledge.adaptation_count})
                if bool(done[0]):break
        _log(out,rows)
        write_json(out/'knowledge.json',{**knowledge.__dict__,
                   'recovered_modalities':sorted(knowledge.recovered_modalities)})
        print(f'MAPE-K execution completed. Results in {out}')
    finally:
        vec.close()

def recover_policy(args):
    """Recover using a previously diagnosed, simulator-ready JSON descriptor."""
    with open(args.descriptor,encoding='utf-8') as stream:
        descriptors=json.load(stream)
    if not isinstance(descriptors,dict) or not descriptors:
        raise ValueError('Descriptor JSON must map sensor names to localized regions')
    raw=make_environment(seed=args.seed,inference_mode=True)
    vec=make_vec_env(raw,training=False)
    try:
        model,out=finetune(raw,vec,base_policy=args.base_policy,descriptors=descriptors,
                           steps=args.finetune_steps,output=args.output,device=args.device,seed=args.seed)
        print(f'Adapted checkpoint saved in {out}')
    finally:
        vec.close()
