import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import mmengine
import torch
from mmengine.dist import init_dist
from mmengine.config import Config, DictAction
from mmengine.utils import get_git_hash

from mmseg import __version__
try:
    from mmengine.runner import set_random_seed
except ImportError:
    # Fallback for older versions or if not available
    from mmcv_custom.train_api import set_random_seed
from mmcv_custom import train_segmentor
from mmseg.models import build_segmentor
from mmengine.logging import MMLogger
# collect_env moved to mmengine.utils in MMSeg 1.x
try:
    from mmengine.utils import collect_env
except ImportError:
    from mmseg.utils import collect_env

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
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

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
        log_level=cfg.log_level
    )

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(model)

    # MMSeg 1.x: Pass dataset config dicts instead of building datasets
    # Runner will handle dataset/dataloader building from configs
    datasets = []
    # Convert to dict if it's a ConfigDict
    if hasattr(cfg.data.train, 'to_dict'):
        train_dataset_cfg = cfg.data.train.to_dict()
    elif hasattr(cfg.data.train, 'copy'):
        train_dataset_cfg = cfg.data.train.copy()
    else:
        train_dataset_cfg = cfg.data.train
    datasets.append(train_dataset_cfg)
    
    # Check if validation should be included
    validate_flag = not args.no_validate
    if validate_flag and hasattr(cfg.data, 'val'):
        # Convert to dict if it's a ConfigDict
        if hasattr(cfg.data.val, 'to_dict'):
            val_dataset_cfg = cfg.data.val.to_dict()
        else:
            val_dataset_cfg = copy.deepcopy(cfg.data.val)
        
        # Ensure validation uses training pipeline
        if isinstance(train_dataset_cfg, dict) and 'pipeline' in train_dataset_cfg:
            val_dataset_cfg['pipeline'] = train_dataset_cfg['pipeline']
        elif hasattr(cfg.data.train, 'pipeline'):
            if isinstance(val_dataset_cfg, dict):
                val_dataset_cfg['pipeline'] = cfg.data.train.pipeline
            else:
                val_dataset_cfg.pipeline = cfg.data.train.pipeline
        datasets.append(val_dataset_cfg)
    
    # Get CLASSES and PALETTE from dataset config for checkpoint metadata
    # Try to get from dataset config, fallback to model if available
    if cfg.checkpoint_config is not None:
        classes = None
        palette = None
        # Try to get from dataset config
        if isinstance(train_dataset_cfg, dict):
            classes = train_dataset_cfg.get('CLASSES', None)
            palette = train_dataset_cfg.get('PALETTE', None)
        elif hasattr(train_dataset_cfg, 'CLASSES'):
            classes = train_dataset_cfg.CLASSES
            palette = getattr(train_dataset_cfg, 'PALETTE', None)
        
        # Fallback to model if available
        if classes is None and hasattr(model, 'CLASSES'):
            classes = model.CLASSES
        if palette is None and hasattr(model, 'PALETTE'):
            palette = model.PALETTE
        
        if classes is not None:
            cfg.checkpoint_config.meta = dict(
                mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                config=cfg.pretty_text,
                CLASSES=classes,
                PALETTE=palette if palette is not None else None
            )
    
    # Set model.CLASSES if available from config
    if isinstance(train_dataset_cfg, dict) and 'CLASSES' in train_dataset_cfg:
        model.CLASSES = train_dataset_cfg['CLASSES']
    elif hasattr(train_dataset_cfg, 'CLASSES'):
        model.CLASSES = train_dataset_cfg.CLASSES
    
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=validate_flag,
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
