"""
MMSeg 1.x / MMEngine Training API

This module provides a simplified training interface that uses Runner.from_cfg()
to handle all model, optimizer, dataloader, and training loop building automatically.

This is the recommended approach for MMSeg 1.x - no manual building of components.
"""

import random
import numpy as np
import torch
from mmengine.runner import Runner


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


def train_segmentor(cfg, distributed=False, validate=False, timestamp=None, meta=None):
    """Launch segmentor training using Runner.from_cfg().
    
    MMSeg 1.x / MMEngine proper approach: Let Runner handle everything.
    No manual building of model, optimizer, dataloaders, etc.
    
    Args:
        cfg: Config object with all required fields (model, train_dataloader,
             optim_wrapper, train_cfg, etc.)
        distributed: Whether to use distributed training (handled by launcher)
        validate: Whether to validate during training
        timestamp: Timestamp string for logging
        meta: Meta dict with additional info
    """
    # Ensure required fields exist
    if not hasattr(cfg, 'work_dir'):
        raise ValueError("Config must have 'work_dir' field")
    
    # Set validation dataloader to None if not validating
    if not validate:
        cfg.val_dataloader = None
        cfg.val_cfg = None
        cfg.val_evaluator = None
    else:
        # Ensure val_evaluator is set when validating
        if not hasattr(cfg, 'val_evaluator') or cfg.val_evaluator is None:
            cfg.val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
    
    # Add meta info if provided
    if meta is not None:
        if not hasattr(cfg, 'meta'):
            cfg.meta = meta
        else:
            cfg.meta.update(meta)
    
    # Use Runner.from_cfg() - the recommended MMEngine approach
    # Runner handles:
    # - Model building from cfg.model
    # - Optimizer/wrapper building from cfg.optim_wrapper (with constructor, paramwise_cfg, etc.)
    # - Dataloader building from cfg.train_dataloader, cfg.val_dataloader
    # - Training loop from cfg.train_cfg
    # - Hooks from cfg.default_hooks, cfg.custom_hooks
    # - Checkpointing, logging, visualization, etc.
    runner = Runner.from_cfg(cfg)
    
    # Set timestamp if provided
    if timestamp is not None:
        runner.timestamp = timestamp
    
    # Start training
    runner.train()
