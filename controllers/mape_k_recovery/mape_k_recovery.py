#!/usr/bin/env python3
"""Webots external controller entry point for MAPE-K targeted policy recovery."""
import argparse

CONDITIONS=("normal","partial_image","full_image","partial_lidar",
            "full_lidar","partial_both","full_both")

def parser():
    top=argparse.ArgumentParser(description="MAPE-K targeted PPO recovery in Webots")
    sub=top.add_subparsers(dest='command',required=True)
    p=sub.add_parser('collect',help='Collect nominal Webots sensor sequences')
    p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=200)
    p.add_argument('--condition',choices=CONDITIONS,default='normal')
    p.add_argument('--steps-per-episode',type=int,default=150)
    p.add_argument('--stride',type=int,default=5)
    p.add_argument('--checkpoint',default=None,help='Optional pretrained PPO navigation policy')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='auto')

    p=sub.add_parser('train-detector',help='Train and calibrate Patch-MIL or LiDAR persistence classifier')
    p.add_argument('--modality',choices=('image','lidar'),required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--device',default='auto')
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--max-train-frames',type=int,default=None)
    p.add_argument('--max-val-frames',type=int,default=None)
    p.add_argument('--initial-checkpoint',default=None,
                   help='Initialize detector finetuning from an existing detector checkpoint')

    p=sub.add_parser('train-policy',help='Train nominal multimodal PPO agent')
    p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=15_000_000)
    p.add_argument('--save-freq',type=int,default=2_000_000)
    p.add_argument('--n-steps',type=int,default=2048)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--device',default='auto')
    p.add_argument('--verbose',type=int,default=1)

    for command in ('infer-policy','infer-mapek','recover-policy'):
        p=sub.add_parser(command)
        p.add_argument('--base-policy',required=True)
        p.add_argument('--output',required=True)
        p.add_argument('--device',default='auto')
        p.add_argument('--seed',type=int,default=4)
        if command!='recover-policy':
            p.add_argument('--condition',choices=CONDITIONS,default='normal')
            p.add_argument('--episodes',type=int,default=5)
            p.add_argument('--max-steps',type=int,default=500)
            p.add_argument('--deterministic',action='store_true',
                           help='Default policy inference is stochastic')
        if command!='infer-policy':
            p.add_argument('--finetune-steps',type=int,default=200_000)
        if command=='infer-mapek':
            p.add_argument('--image-detector',required=True)
            p.add_argument('--lidar-detector',required=True)
            p.add_argument('--onset',type=int,default=5)
            p.add_argument('--max-recoveries',type=int,default=1)
            p.add_argument('--cooldown',type=int,default=100)
            p.add_argument('--scan-degrees',type=int,default=360)
            p.add_argument('--scan-capture-every',type=int,default=5)
            p.add_argument('--image-stability-tol',type=float,default=2.0)
            p.add_argument('--image-morph-kernel',type=int,default=5)
            p.add_argument('--image-min-component-area',type=int,default=500)
            p.add_argument('--image-max-component-fraction',type=float,default=.60)
            p.add_argument('--full-region-fraction',type=float,default=.90)
            p.add_argument('--lidar-stability-eps',type=float,default=.01)
            p.add_argument('--lidar-value-min',type=float,default=.95)
            p.add_argument('--lidar-min-width',type=int,default=20)
            p.add_argument('--lidar-relaxed-percentile',type=float,default=2.0)
        if command=='recover-policy':
            p.add_argument('--descriptor',required=True,
                           help='JSON with image and/or lidar localized descriptor')
    p=sub.add_parser('infer-detector',help='Print calibrated learned-detector scores in Webots')
    p.add_argument('--image-detector',required=True)
    p.add_argument('--lidar-detector',required=True)
    p.add_argument('--condition',choices=CONDITIONS,default='normal')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seed',type=int,default=4)
    p.add_argument('--device',default='auto')
    return top

def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='collect':
        from recovery.dataset import collect as run
    elif args.command=='train-detector':
        from recovery.train_detector import run
        if args.batch_size is None:args.batch_size=4 if args.modality=='image' else 32
    elif args.command=='train-policy':
        from recovery.train_policy import run
    elif args.command=='infer-policy':
        from recovery.infer import infer_policy as run
    elif args.command=='infer-mapek':
        from recovery.infer import infer_mapek as run
    elif args.command=='recover-policy':
        from recovery.infer import recover_policy as run
    else:
        from recovery.detect import run
    return run(args)

if __name__=='__main__':main()
