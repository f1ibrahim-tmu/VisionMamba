"""
MMSeg 1.x Training Script

Uses Runner.from_cfg() to handle all model, optimizer, dataloader,
and training loop building automatically.
"""

import argparse
import os
import os.path as osp
import time

import mmengine
import torch
from mmengine.dist import init_dist
from mmengine.config import Config, DictAction
from mmengine.utils import get_git_hash
from mmengine.logging import MMLogger

from mmseg import __version__
try:
    from mmengine.runner import set_random_seed
except ImportError:
    from mmcv_custom.train_api import set_random_seed

from mmcv_custom import train_segmentor
# Import custom optimizer constructor to register it
from mmcv_custom import VimLayerDecayOptimizerConstructor

# collect_env moved to mmengine.utils in MMSeg 1.x
try:
    from mmengine.utils import collect_env
except ImportError:
    from mmseg.utils import collect_env

# Import backbone to register it with MMSeg registry
from backbone import vim


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # Weights & Biases arguments
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', default='visionmamba', type=str, help='W&B project name')
    parser.add_argument('--wandb-entity', default=None, type=str, help='W&B entity/team name')
    parser.add_argument('--wandb-run-name', default=None, type=str, help='W&B run name')
    parser.add_argument('--wandb-tags', nargs='+', default=[], help='Tags for W&B run')
    
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    
    # Handle load_from and resume_from
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    
    # Set gpu_ids for compatibility
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        cfg.launcher = 'none'
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_params', {}))
        cfg.launcher = args.launcher

    # create work_dir
    mmengine.utils.mkdir_or_exist(osp.abspath(cfg.work_dir))
    
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = MMLogger.get_instance(
        name='mmseg',
        log_file=log_file,
        log_level=cfg.get('log_level', 'INFO')
    )

    # init the meta dict to record some important information
    meta = dict()
    
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # Ensure default_scope is set for registry
    if not hasattr(cfg, 'default_scope'):
        cfg.default_scope = 'mmseg'
    
    # Ensure env_cfg is set for Runner
    if not hasattr(cfg, 'env_cfg'):
        cfg.env_cfg = dict(
            cudnn_benchmark=cfg.get('cudnn_benchmark', False),
            mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
            dist_cfg=dict(backend='nccl')
        )
    
    # Ensure randomness config
    if not hasattr(cfg, 'randomness'):
        cfg.randomness = dict(seed=args.seed, deterministic=args.deterministic)
    
    # Add W&B config to cfg
    if args.use_wandb:
        cfg.use_wandb = True
        cfg.wandb_project = args.wandb_project
        cfg.wandb_entity = args.wandb_entity
        cfg.wandb_run_name = args.wandb_run_name
        cfg.wandb_tags = args.wandb_tags
    
    # Run training using Runner.from_cfg()
    # This handles everything: model, optimizer, dataloaders, training loop
    validate_flag = not args.no_validate
    
    train_segmentor(
        cfg,
        distributed=distributed,
        validate=validate_flag,
        timestamp=timestamp,
        meta=meta
    )


if __name__ == '__main__':
    main()
