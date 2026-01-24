#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys

# Ensure we import from local det/detectron2 directory
# Get the directory containing this script (det/tools/)
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the det/ directory (parent of tools/)
_det_dir = os.path.dirname(_script_dir)
# Add det/ to sys.path at the beginning so local detectron2 takes precedence
if _det_dir not in sys.path:
    sys.path.insert(0, _det_dir)

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
import torch
import argparse

logger = logging.getLogger("detectron2")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )
    
    # Register WandB hook if enabled
    hook_list = [
        hooks.IterationTimer(),
        hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
        hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
        if comm.is_main_process()
        else None,
        hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
        hooks.PeriodicWriter(
            default_writers(cfg.train.output_dir, cfg.train.max_iter),
            period=cfg.train.log_period,
        )
        if comm.is_main_process()
        else None,
    ]
    
    # Add WandB hook if enabled
    if args and getattr(args, 'use_wandb', False):
        try:
            from detectron2.engine.hooks import WandbHook
            wandb_hook = WandbHook(
                project=getattr(args, 'wandb_project', 'detectron2'),
                entity=getattr(args, 'wandb_entity', None),
                name=getattr(args, 'wandb_run_name', None),
                tags=getattr(args, 'wandb_tags', []),
                enabled=True
            )
            # Attach trainer to hook (needed for hook callbacks)
            wandb_hook.trainer = trainer
            hook_list.append(wandb_hook if comm.is_main_process() else None)
            logger.info("WandB hook registered: project=%s, name=%s", 
                       getattr(args, 'wandb_project', 'detectron2'),
                       getattr(args, 'wandb_run_name', None))
        except ImportError:
            logger.warning("WandB not available. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"Failed to register WandB hook: {e}")
    
    trainer.register_hooks(hook_list)

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    # If running from project root and config path doesn't start with det/,
    # try to find it in det/projects/ first
    import os
    config_file = args.config_file
    if not os.path.isabs(config_file) and not config_file.startswith("det/"):
        # Check if we're in project root (current dir has det/ subdirectory)
        cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get project root: script is at det/tools/lazyconfig_train_net.py
        project_root = os.path.dirname(os.path.dirname(script_dir))
        
        # If current working directory is project root, and config path starts with projects/
        if cwd == project_root and config_file.startswith("projects/"):
            # Try det/projects/ path first
            det_config_path = os.path.join(project_root, "det", config_file)
            if os.path.exists(det_config_path):
                config_file = os.path.join("det", config_file)
        # If current working directory is det/, keep original path
        elif os.path.basename(cwd) == "det" and config_file.startswith("projects/"):
            # Already correct, keep as is
            pass
    
    # Update args.config_file so default_setup can find it
    args.config_file = config_file
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    parser = default_argument_parser()
    # Add WandB arguments
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', default='visionmamba', type=str, help='W&B project name')
    parser.add_argument('--wandb-entity', default=None, type=str, help='W&B entity/team name')
    parser.add_argument('--wandb-run-name', default=None, type=str, help='W&B run name')
    parser.add_argument('--wandb-tags', nargs='+', default=[], help='Tags for W&B run')
    args = parser.parse_args()
    
    # Check if already in distributed environment (from torch.distributed.run)
    # When torch.distributed.run spawns processes, it sets RANK and WORLD_SIZE
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Already launched by torch.distributed.run, skip launch() to avoid double spawning
        # Need to set CUDA device and create local process group
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # Set CUDA device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.info("Set CUDA device to local_rank=%d", local_rank)
        
        # Create local process group (needed for Detectron2)
        # For single node, num_gpus_per_machine equals world_size
        num_gpus_per_machine = world_size
        comm.create_local_process_group(num_gpus_per_machine)
        
        logger.info("Detected existing distributed environment (RANK=%s, WORLD_SIZE=%s, LOCAL_RANK=%s), "
                   "skipping Detectron2 launch()", os.environ.get("RANK"), world_size, local_rank)
        main(args)
    else:
        # Not in distributed environment, use Detectron2's launch to spawn processes
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
