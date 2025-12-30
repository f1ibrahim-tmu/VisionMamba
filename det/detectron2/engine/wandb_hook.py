# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

import os
import logging
from typing import Optional

import detectron2.utils.comm as comm
from .train_loop import HookBase

logger = logging.getLogger("detectron2")

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class WandbHook(HookBase):
    """
    Hook to log metrics to Weights & Biases.
    """

    def __init__(
        self,
        project: Optional[str] = "detectron2",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        """
        Args:
            project: W&B project name
            entity: W&B entity/team name
            name: W&B run name
            tags: List of tags for the run
            config: Additional config to log
            enabled: Whether to enable W&B logging
        """
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.config = config or {}
        self.enabled = enabled and WANDB_AVAILABLE
        self._initialized = False

    def before_train(self):
        """Initialize W&B before training starts."""
        if not self.enabled:
            return

        if not comm.is_main_process():
            return

        if self._initialized:
            return

        try:
            # Get run name from output dir if not provided
            if self.name is None:
                output_dir = self.trainer.cfg.OUTPUT_DIR if hasattr(self.trainer, 'cfg') else None
                if output_dir:
                    self.name = os.path.basename(output_dir.rstrip('/'))
                else:
                    self.name = "detectron2-run"

            # Merge config from trainer
            config = self.config.copy()
            if hasattr(self.trainer, 'cfg'):
                # Log key config parameters
                cfg = self.trainer.cfg
                config.update({
                    'model': cfg.MODEL.META_ARCHITECTURE if hasattr(cfg.MODEL, 'META_ARCHITECTURE') else None,
                    'batch_size': cfg.SOLVER.IMS_PER_BATCH if hasattr(cfg.SOLVER, 'IMS_PER_BATCH') else None,
                    'max_iter': cfg.SOLVER.MAX_ITER if hasattr(cfg.SOLVER, 'MAX_ITER') else None,
                    'base_lr': cfg.SOLVER.BASE_LR if hasattr(cfg.SOLVER, 'BASE_LR') else None,
                    'weight_decay': cfg.SOLVER.WEIGHT_DECAY if hasattr(cfg.SOLVER, 'WEIGHT_DECAY') else None,
                })

            wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                tags=self.tags if self.tags else None,
                config=config,
                reinit=True,
                settings=wandb.Settings(start_method="fork" if hasattr(wandb.Settings, 'start_method') else None)
            )
            self._initialized = True
            logger.info(f"W&B initialized: project={self.project}, name={self.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
            self.enabled = False

    def after_step(self):
        """Log metrics after each step."""
        if not self.enabled or not self._initialized:
            return

        if not comm.is_main_process():
            return

        try:
            storage = self.trainer.storage
            if storage is None:
                return

            # Get latest metrics from storage
            latest = storage.latest()
            if not latest:
                return

            # Log to W&B
            wandb_log = {}
            for k, (v, iter) in latest.items():
                if isinstance(v, (int, float)):
                    wandb_log[f"train/{k}"] = v

            if wandb_log:
                wandb.log(wandb_log, step=storage.iter)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def after_train(self):
        """Finalize W&B run after training."""
        if not self.enabled or not self._initialized:
            return

        if not comm.is_main_process():
            return

        try:
            wandb.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")

