import random
import warnings

import numpy as np
import torch
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner
# MMSegmentation 1.0.0+ moved eval hooks to mmseg.engine
try:
    from mmseg.engine import SegEvaluator
except ImportError:
    SegEvaluator = None
from mmengine.logging import MMLogger


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
    """Launch segmentor training.
    
    Args:
        model: The model to train
        dataset: Dataset config dict(s) or None. If None, expects cfg.train_dataloader/val_dataloader.
        cfg: Config object
        distributed: Whether to use distributed training
        validate: Whether to validate during training
        timestamp: Timestamp string
        meta: Meta dict
    """
    logger = MMLogger.get_instance(
        name='mmseg',
        log_level=cfg.log_level
    )

    # MMSeg 1.x: Let Runner handle all dataset/dataloader building from config dicts
    # Convert dataset configs to dataloader configs if needed
    train_dataloader_cfg = None
    val_dataloader_cfg = None
    
    # Check if explicit dataloader configs exist (preferred MMSeg 1.x format)
    if hasattr(cfg, 'train_dataloader') and cfg.train_dataloader is not None:
        train_dataloader_cfg = cfg.train_dataloader.copy()
    elif dataset is not None:
        # Convert dataset config(s) to dataloader config(s)
        dataset_list = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        train_dataset_cfg = dataset_list[0]
        
        # Convert to dataloader config dict
        train_dataloader_cfg = dict(
            dataset=train_dataset_cfg if isinstance(train_dataset_cfg, dict) else train_dataset_cfg,
            batch_size=cfg.data.samples_per_gpu if hasattr(cfg.data, 'samples_per_gpu') else 4,
            num_workers=cfg.data.workers_per_gpu if hasattr(cfg.data, 'workers_per_gpu') else 4,
            sampler=dict(type='InfiniteSampler', shuffle=True),
            drop_last=True
        )
        
        # Handle validation dataloader
        if validate and len(dataset_list) > 1:
            val_dataset_cfg = dataset_list[1]
            val_dataloader_cfg = dict(
                dataset=val_dataset_cfg if isinstance(val_dataset_cfg, dict) else val_dataset_cfg,
                batch_size=1,
                num_workers=cfg.data.workers_per_gpu if hasattr(cfg.data, 'workers_per_gpu') else 4,
                sampler=dict(type='DefaultSampler', shuffle=False),
                drop_last=False
            )
    elif hasattr(cfg, 'val_dataloader') and cfg.val_dataloader is not None:
        val_dataloader_cfg = cfg.val_dataloader.copy()
    
    # Ensure samplers are set if not provided
    if train_dataloader_cfg is not None and 'sampler' not in train_dataloader_cfg:
        train_dataloader_cfg['sampler'] = dict(type='InfiniteSampler', shuffle=True)
    if val_dataloader_cfg is not None and 'sampler' not in val_dataloader_cfg:
        val_dataloader_cfg['sampler'] = dict(type='DefaultSampler', shuffle=False)

    # build optimizer - MMEngine uses optim_wrapper
    def remove_delete_keys(d):
        """Recursively remove _delete_ keys from dict (config inheritance artifact)."""
        if isinstance(d, dict):
            d.pop('_delete_', None)
            for v in d.values():
                remove_delete_keys(v)
        return d
    
    if hasattr(cfg, 'optim_wrapper'):
        # Deep copy and convert to dict, then remove _delete_ keys
        import copy
        optim_wrapper_cfg = copy.deepcopy(cfg.optim_wrapper)
        if hasattr(optim_wrapper_cfg, 'to_dict'):
            optim_wrapper_cfg = optim_wrapper_cfg.to_dict()
        elif not isinstance(optim_wrapper_cfg, dict):
            optim_wrapper_cfg = dict(optim_wrapper_cfg)
        
        # Recursively remove _delete_ from all nested dicts
        remove_delete_keys(optim_wrapper_cfg)
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

    # MMEngine ≥ 0.10: MMDataParallel and MMDistributedDataParallel are removed
    # Runner handles model wrapping automatically based on distributed setting
    # We just need to move model to device - Runner will handle the rest
    if torch.cuda.is_available():
        if distributed:
            # For distributed training, Runner will wrap with DDP automatically
            # Just ensure model is on the correct device
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
            model = model.to(device)
        else:
            # For single GPU or DataParallel, move to first GPU
            device = torch.device(f'cuda:{cfg.gpu_ids[0]}')
            model = model.to(device)

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
    
    # Create Runner with new API - pass config dicts, Runner will build dataloaders automatically
    runner = Runner(
        model=model,
        optim_wrapper=optim_wrapper,
        work_dir=cfg.work_dir,
        train_dataloader=train_dataloader_cfg,
        val_dataloader=val_dataloader_cfg if validate else None,
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
