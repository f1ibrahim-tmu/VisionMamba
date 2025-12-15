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

    # Prepare data loaders - support both MMEngine format and legacy format
    train_dataloader = None
    val_dataloader = None
    
    # Check if explicit dataloader configs exist (MMEngine format)
    if hasattr(cfg, 'train_dataloader') and cfg.train_dataloader is not None:
        # Use explicit dataloader config - Runner will build it automatically
        # But we need to ensure dataset is included if not in config
        train_dataloader_cfg = cfg.train_dataloader.copy()
        
        # If dataset is not in config, use the dataset passed as argument
        if 'dataset' not in train_dataloader_cfg:
            train_dataset = dataset[0] if isinstance(dataset, (list, tuple)) else dataset
            # Build dataset if it's a dict, otherwise use directly
            if isinstance(train_dataset, dict):
                train_dataloader_cfg['dataset'] = build_dataset(train_dataset)
            else:
                train_dataloader_cfg['dataset'] = train_dataset
        elif isinstance(train_dataloader_cfg['dataset'], dict):
            # Build dataset from dict config
            train_dataloader_cfg['dataset'] = build_dataset(train_dataloader_cfg['dataset'])
        
        # Runner can build from config, but we can also build it here for compatibility
        # Build dataloader manually to ensure compatibility with mmseg's build_dataloader
        train_dataloader = build_dataloader(
            train_dataloader_cfg['dataset'],
            train_dataloader_cfg.get('batch_size', 4),
            train_dataloader_cfg.get('num_workers', 4),
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            drop_last=train_dataloader_cfg.get('drop_last', True)
        )
        
        if validate and hasattr(cfg, 'val_dataloader') and cfg.val_dataloader is not None:
            val_dataloader_cfg = cfg.val_dataloader.copy()
            if 'dataset' not in val_dataloader_cfg:
                val_dataset = dataset[1] if (isinstance(dataset, (list, tuple)) and len(dataset) > 1) else None
                if val_dataset is not None:
                    if isinstance(val_dataset, dict):
                        val_dataloader_cfg['dataset'] = build_dataset(val_dataset)
                    else:
                        val_dataloader_cfg['dataset'] = val_dataset
            elif isinstance(val_dataloader_cfg['dataset'], dict):
                val_dataloader_cfg['dataset'] = build_dataset(val_dataloader_cfg['dataset'])
            
            if 'dataset' in val_dataloader_cfg:
                val_dataloader = build_dataloader(
                    val_dataloader_cfg['dataset'],
                    val_dataloader_cfg.get('batch_size', 1),
                    val_dataloader_cfg.get('num_workers', 4),
                    len(cfg.gpu_ids),
                    dist=distributed,
                    seed=cfg.seed,
                    drop_last=False
                )
    else:
        # Legacy format: build from dataset objects and samples_per_gpu
        dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu if hasattr(cfg.data, 'samples_per_gpu') else 4,
                cfg.data.workers_per_gpu if hasattr(cfg.data, 'workers_per_gpu') else 4,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed,
                drop_last=True) for ds in dataset
        ]
        train_dataloader = data_loaders[0]
        val_dataloader = data_loaders[1] if len(data_loaders) > 1 and validate else None

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
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader if validate else None,
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
