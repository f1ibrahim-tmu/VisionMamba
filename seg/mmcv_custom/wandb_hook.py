# -*- coding: utf-8 -*-
import os
import logging
from typing import Optional

from mmcv.runner import HOOKS, Hook

logger = logging.getLogger(__name__)

# Optional W&B import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@HOOKS.register_module()
class WandbHook(Hook):
    """Hook to log metrics to Weights & Biases for MMSegmentation."""

    def __init__(
        self,
        project: Optional[str] = "mmsegmentation",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
        log_interval: int = 50,
    ):
        """
        Args:
            project: W&B project name
            entity: W&B entity/team name
            name: W&B run name
            tags: List of tags for the run
            config: Additional config to log
            enabled: Whether to enable W&B logging
            log_interval: Interval to log metrics (in iterations)
        """
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.config = config or {}
        self.enabled = enabled and WANDB_AVAILABLE
        self.log_interval = log_interval
        self._initialized = False

    def before_run(self, runner):
        """Initialize W&B before training starts."""
        if not self.enabled:
            return

        if not self._initialized:
            try:
                # Get run name from work_dir if not provided
                if self.name is None:
                    work_dir = runner.work_dir if hasattr(runner, 'work_dir') else None
                    if work_dir:
                        self.name = os.path.basename(work_dir.rstrip('/'))
                    else:
                        self.name = "mmseg-run"

                # Merge config from runner
                config = self.config.copy()
                if hasattr(runner, 'cfg'):
                    cfg = runner.cfg
                    config.update({
                        'model': cfg.model.type if hasattr(cfg, 'model') else None,
                        'batch_size': cfg.data.samples_per_gpu if hasattr(cfg, 'data') else None,
                        'max_iters': cfg.runner.max_iters if hasattr(cfg, 'runner') and hasattr(cfg.runner, 'max_iters') else None,
                        'lr': cfg.optimizer.lr if hasattr(cfg, 'optimizer') else None,
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

    def after_train_iter(self, runner):
        """Log metrics after each training iteration."""
        if not self.enabled or not self._initialized:
            return

        if runner.iter % self.log_interval != 0:
            return

        try:
            # Get metrics from runner's logger
            if hasattr(runner, 'logger') and hasattr(runner.logger, 'log_dict'):
                # Try to get metrics from the logger
                wandb_log = {}
                
                # Common metrics in MMSegmentation
                if hasattr(runner, 'outputs') and runner.outputs:
                    outputs = runner.outputs
                    if isinstance(outputs, dict):
                        if 'log_vars' in outputs:
                            for k, v in outputs['log_vars'].items():
                                if isinstance(v, (int, float)):
                                    wandb_log[f"train/{k}"] = v
                
                if wandb_log:
                    wandb.log(wandb_log, step=runner.iter)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")

    def after_val_iter(self, runner):
        """Log validation metrics after each validation iteration."""
        if not self.enabled or not self._initialized:
            return
        # Validation metrics are typically logged in after_val_epoch
        pass

    def after_val_epoch(self, runner):
        """Log validation metrics after each validation epoch."""
        if not self.enabled or not self._initialized:
            return

        try:
            # Get validation metrics from runner
            wandb_log = {}
            if hasattr(runner, 'logger') and hasattr(runner.logger, 'log_dict'):
                # Try to extract validation metrics
                if hasattr(runner, 'outputs') and runner.outputs:
                    outputs = runner.outputs
                    if isinstance(outputs, dict):
                        if 'log_vars' in outputs:
                            for k, v in outputs['log_vars'].items():
                                if isinstance(v, (int, float)):
                                    wandb_log[f"val/{k}"] = v

            if wandb_log:
                wandb.log(wandb_log, step=runner.iter)
        except Exception as e:
            logger.warning(f"Failed to log validation metrics to W&B: {e}")

    def after_run(self, runner):
        """Finalize W&B run after training."""
        if not self.enabled or not self._initialized:
            return

        try:
            wandb.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.warning(f"Failed to finish W&B run: {e}")

