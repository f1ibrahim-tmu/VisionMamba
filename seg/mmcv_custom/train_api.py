import random
import warnings

import numpy as np
import torch
from mmengine.model import MMDataParallel, MMDistributedDataParallel
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
# MMSegmentation 1.0.0+ moved eval hooks to mmseg.engine
try:
    from mmseg.engine import SegEvaluator
except ImportError:
    SegEvaluator = None
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_segmentor(model,
                    dataset,
                    cfg,
                    distributed=False,
                    validate=False,
                    timestamp=None,
                    meta=None):
    """Launch segmentor training."""
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=True) for ds in dataset
    ]

    # build optimizer - MMEngine uses optim_wrapper
    if hasattr(cfg, 'optim_wrapper'):
        optim_wrapper_cfg = cfg.optim_wrapper
    else:
        # Convert old optimizer config to optim_wrapper format
        optim_wrapper_cfg = dict(
            type='OptimWrapper',
            optimizer=cfg.optimizer,
            clip_grad=cfg.optimizer_config.get('grad_clip', None) if hasattr(cfg, 'optimizer_config') else None
        )
        # Handle FP16 - use native PyTorch AMP (AmpOptimWrapper)
        if hasattr(cfg, 'optimizer_config') and cfg.optimizer_config.get("use_fp16", False):
            optim_wrapper_cfg['type'] = 'AmpOptimWrapper'
    
    optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # MMEngine ≥ 0.7 uses Runner with train_cfg/val_cfg/test_cfg
    # Convert old runner config to new format
    max_iters = cfg.total_iters if hasattr(cfg, 'total_iters') else 60000
    if hasattr(cfg, 'runner') and cfg.runner is not None:
        # Extract max_iters from old runner config if present
        if 'max_iters' in cfg.runner:
            max_iters = cfg.runner['max_iters']
    
    # Set up train_cfg for iteration-based training
    train_cfg = dict(
        type='IterBasedTrainLoop',
        max_iters=max_iters,
        val_interval=cfg.evaluation.get('interval', 1000) if hasattr(cfg, 'evaluation') else 1000
    )
    
    # Set up val_cfg
    val_cfg = dict(type='ValLoop')
    
    # Set up test_cfg
    test_cfg = dict(type='TestLoop')
    
    # Get default_hooks from config, or use None (Runner will use defaults)
    default_hooks = cfg.get('default_hooks', None)
    
    # Create Runner with new API
    runner = Runner(
        model=model,
        optim_wrapper=optim_wrapper,
        work_dir=cfg.work_dir,
        train_dataloader=data_loaders[0],
        val_dataloader=data_loaders[1] if len(data_loaders) > 1 and validate else None,
        train_cfg=train_cfg,
        val_cfg=val_cfg,
        test_cfg=test_cfg,
        default_hooks=default_hooks,
        custom_hooks=cfg.get('custom_hooks', None),
        default_scope=cfg.get('default_scope', 'mmseg'),
        param_scheduler=cfg.get('param_scheduler', None),
        logger=logger,
        log_level=cfg.log_level,
        meta=meta
    )

    # Register custom hooks if needed
    # Note: Most hooks should be configured via default_hooks in config
    # But we can register custom hooks here if needed
    from .throughput_hook import ThroughputHook
    log_interval = cfg.log_config.get('interval', 50) if cfg.log_config else 50
    runner.register_hook(ThroughputHook(log_interval=log_interval), priority='NORMAL')

    # Set timestamp for log file naming
    runner.timestamp = timestamp

    # Handle resume/load checkpoint
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    # Start training
    runner.train()
