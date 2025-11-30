import random
import warnings

import numpy as np
import torch
from mmengine.model import MMDataParallel, MMDistributedDataParallel
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner

# MMSegmentation 1.0.0+ moved eval hooks to mmseg.engine
try:
    from mmseg.engine import DistEvalHook, EvalHook
except ImportError:
    # Fallback for older versions
    from mmseg.core import DistEvalHook, EvalHook
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.utils import get_root_logger
try:
    import apex
except:
    print('apex is not installed')


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

    # build optimizer - MMCV 2.x uses optim_wrapper instead of optimizer
    # For backward compatibility, we'll build optimizer from cfg.optimizer
    if hasattr(cfg, 'optim_wrapper'):
        optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
        optimizer = optim_wrapper.optimizer
    else:
        # Fallback to old API if optim_wrapper not available
        from mmengine.optim import build_optim_wrapper
        optim_wrapper_cfg = dict(type='OptimWrapper', optimizer=cfg.optimizer)
        optim_wrapper = build_optim_wrapper(model, optim_wrapper_cfg)
        optimizer = optim_wrapper.optimizer

    # use apex fp16 optimizer
    if cfg.optimizer_config.get("type", None) and cfg.optimizer_config["type"] == "DistOptimizerHook":
        if cfg.optimizer_config.get("use_fp16", False):
            model, optimizer = apex.amp.initialize(
                model.cuda(), optimizer, opt_level="O1")
            for m in model.modules():
                if hasattr(m, "fp16_enabled"):
                    m.fp16_enabled = True

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

    # MMCV 2.x uses Runner directly, not build_runner
    # Check if runner config exists, otherwise create default
    if cfg.get('runner') is None:
        cfg.runner = {'type': 'IterBasedRunner', 'max_iters': cfg.total_iters}
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)

    # For MMCV 2.x, we need to use Runner class directly
    # Import the appropriate runner type
    runner_type = cfg.runner.get('type', 'IterBasedRunner')
    if runner_type == 'IterBasedRunner':
        from mmengine.runner import IterBasedRunner
        RunnerClass = IterBasedRunner
    else:
        from mmengine.runner import EpochBasedRunner
        RunnerClass = EpochBasedRunner
    
    runner = RunnerClass(
        model=model,
        optim_wrapper=optim_wrapper,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta,
        **{k: v for k, v in cfg.runner.items() if k != 'type'})

    # register hooks - MMCV 2.x uses different hook registration
    # Note: In MMCV 2.x, hooks are typically registered via default_hooks in config
    # But for backward compatibility, we'll try to register them manually
    # The runner should handle default hooks automatically if configured properly
    # Custom hooks can still be registered here if needed
    
    # register throughput hook
    from .throughput_hook import ThroughputHook
    log_interval = cfg.log_config.get('interval', 50) if cfg.log_config else 50
    runner.register_hook(ThroughputHook(log_interval=log_interval), priority='NORMAL')

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp

    # register eval hooks - MMCV 2.x uses different eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = 'IterBasedRunner' not in cfg.runner.get('type', 'IterBasedRunner')
        # MMCV 2.x uses different eval hooks from mmseg
        from mmseg.engine import SegEvaluator
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    # MMCV 2.x uses train_dataloader and val_dataloader instead of workflow
    # Set dataloaders before resuming/loading
    runner.train_dataloader = data_loaders[0]
    if len(data_loaders) > 1:
        runner.val_dataloader = data_loaders[1]
    
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    
    # Start training - MMCV 2.x uses runner.train() directly
    runner.train()
