# -*- coding: utf-8 -*-
import time
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class ThroughputHook(Hook):
    """Hook to log throughput metrics (img/sec and samples/epoch)."""

    def __init__(self, log_interval=50):
        """
        Args:
            log_interval (int): Interval to log throughput metrics.
        """
        self.log_interval = log_interval
        self.iter_start_time = None
        self.epoch_start_time = None
        self.total_samples = 0
        self.epoch_samples = 0

    def before_train_iter(self, runner):
        """Called before each training iteration."""
        self.iter_start_time = time.time()

    def after_train_iter(self, runner):
        """Called after each training iteration."""
        if self.iter_start_time is None:
            return

        # Get batch size from the data batch
        batch_size = 1
        if hasattr(runner, 'data_batch'):
            batch = runner.data_batch
            if isinstance(batch, dict):
                # For segmentation, images are usually in 'img' key
                if 'img' in batch:
                    batch_size = batch['img'].shape[0] if hasattr(batch['img'], 'shape') else 1
                elif 'inputs' in batch:
                    batch_size = batch['inputs'].shape[0] if hasattr(batch['inputs'], 'shape') else 1
                else:
                    # Try to get first tensor's batch size
                    for v in batch.values():
                        if hasattr(v, 'shape') and len(v.shape) > 0:
                            batch_size = v.shape[0]
                            break
            elif isinstance(batch, (list, tuple)):
                if len(batch) > 0 and hasattr(batch[0], 'shape'):
                    batch_size = batch[0].shape[0]
        else:
            # Fallback: try to get from config
            try:
                batch_size = runner.cfg.data.samples_per_gpu
            except:
                batch_size = 1

        # Calculate iteration time and throughput
        iter_time = time.time() - self.iter_start_time
        if iter_time > 0:
            img_per_sec = batch_size / iter_time
            # Log to runner's logger
            runner.logger.info(f'Throughput: {img_per_sec:.2f} img/s')
            
            # Store in runner's log buffer for periodic logging
            if not hasattr(runner, 'throughput_buffer'):
                runner.throughput_buffer = []
            runner.throughput_buffer.append({
                'img_per_sec': img_per_sec,
                'batch_size': batch_size,
                'iter_time': iter_time
            })

        # Track samples per epoch
        self.epoch_samples += batch_size
        self.total_samples += batch_size

    def before_train_epoch(self, runner):
        """Called before each training epoch."""
        self.epoch_start_time = time.time()
        self.epoch_samples = 0
        if hasattr(runner, 'throughput_buffer'):
            runner.throughput_buffer = []

    def after_train_epoch(self, runner):
        """Called after each training epoch."""
        if self.epoch_start_time is None:
            return

        epoch_time = time.time() - self.epoch_start_time
        
        # Calculate average throughput for the epoch
        if hasattr(runner, 'throughput_buffer') and len(runner.throughput_buffer) > 0:
            avg_img_per_sec = sum(x['img_per_sec'] for x in runner.throughput_buffer) / len(runner.throughput_buffer)
        else:
            avg_img_per_sec = 0.0

        # Log epoch-level metrics
        runner.logger.info(
            f'Epoch [{runner.epoch}] - Throughput: {avg_img_per_sec:.2f} img/s, '
            f'Samples per epoch: {self.epoch_samples}, Epoch time: {epoch_time:.2f}s'
        )

        # Add to runner's log dict for JSON logging
        if not hasattr(runner, 'log_dict'):
            runner.log_dict = {}
        runner.log_dict['img_per_sec'] = avg_img_per_sec
        runner.log_dict['samples_per_epoch'] = self.epoch_samples
        runner.log_dict['epoch_time'] = epoch_time

