# -*- coding: utf-8 -*-
"""Throughput Hook for MMEngine Runner.

This hook logs throughput metrics (img/sec) during training.
Compatible with MMEngine 0.10+ Hook API.
"""
import time
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class ThroughputHook(Hook):
    """Hook to log throughput metrics (img/sec).
    
    MMEngine 0.10+ compatible hook that logs training throughput.
    """

    def __init__(self, log_interval=50):
        """
        Args:
            log_interval (int): Interval (in iterations) to log throughput metrics.
        """
        self.log_interval = log_interval
        self.iter_start_time = None
        self.total_samples = 0
        self.iter_samples = []
        self.iter_times = []

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        """Called before each training iteration.
        
        Args:
            runner: The runner object
            batch_idx: Current batch index
            data_batch: The data batch (may be None)
        """
        self.iter_start_time = time.time()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Called after each training iteration.
        
        Args:
            runner: The runner object
            batch_idx: Current batch index
            data_batch: The data batch
            outputs: The model outputs
        """
        if self.iter_start_time is None:
            return

        # Get batch size from the data batch
        batch_size = self._get_batch_size(data_batch, runner)

        # Calculate iteration time
        iter_time = time.time() - self.iter_start_time
        
        # Track for averaging
        self.iter_samples.append(batch_size)
        self.iter_times.append(iter_time)
        self.total_samples += batch_size

        # Log at specified interval
        current_iter = runner.iter + 1
        if current_iter % self.log_interval == 0 and len(self.iter_times) > 0:
            # Calculate average throughput over the interval
            total_time = sum(self.iter_times)
            total_samples = sum(self.iter_samples)
            if total_time > 0:
                avg_img_per_sec = total_samples / total_time
                runner.logger.info(f'Throughput: {avg_img_per_sec:.2f} img/s (avg over {len(self.iter_times)} iters)')
            
            # Reset for next interval
            self.iter_samples = []
            self.iter_times = []

    def _get_batch_size(self, data_batch, runner):
        """Extract batch size from data batch.
        
        Args:
            data_batch: The data batch dict or tuple
            runner: The runner object
            
        Returns:
            int: The batch size
        """
        batch_size = 1
        
        if data_batch is None:
            # Try to get from runner's data_batch attribute
            if hasattr(runner, 'data_batch'):
                data_batch = runner.data_batch
        
        if data_batch is not None:
            if isinstance(data_batch, dict):
                # MMEngine format: 'inputs' key contains the images
                if 'inputs' in data_batch:
                    inputs = data_batch['inputs']
                    if hasattr(inputs, 'shape'):
                        batch_size = inputs.shape[0]
                    elif isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                        if hasattr(inputs[0], 'shape'):
                            batch_size = len(inputs)
                # Legacy format: 'img' key
                elif 'img' in data_batch:
                    img = data_batch['img']
                    if hasattr(img, 'shape'):
                        batch_size = img.shape[0]
                # Try first tensor
                else:
                    for v in data_batch.values():
                        if hasattr(v, 'shape') and len(v.shape) > 0:
                            batch_size = v.shape[0]
                            break
            elif isinstance(data_batch, (list, tuple)):
                if len(data_batch) > 0:
                    if hasattr(data_batch[0], 'shape'):
                        batch_size = data_batch[0].shape[0]
                    else:
                        batch_size = len(data_batch)
        
        return batch_size
