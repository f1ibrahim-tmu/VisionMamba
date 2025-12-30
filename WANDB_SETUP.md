# Weights & Biases (W&B) Setup Guide

This guide explains how to set up and use Weights & Biases (W&B) for experiment tracking and metrics analysis in the VisionMamba project.

## What is Weights & Biases?

Weights & Biases (W&B) is a platform for experiment tracking, model versioning, and performance visualization. It helps you:

- **Track training metrics** (loss, accuracy, learning rate) in real-time
- **Compare experiments** across different hyperparameters
- **Monitor system resources** (GPU usage, memory, CPU)
- **Log model checkpoints** and artifacts
- **Visualize results** with interactive dashboards
- **Collaborate** with team members on experiments

## Installation

1. Install the W&B Python package:
```bash
pip install wandb
```

2. Login to W&B using your API key:
```bash
wandb login 964b6fd194cced2fc3fabd18754dd54218145929
```

Alternatively, you can set the API key as an environment variable:
```bash
export WANDB_API_KEY=964b6fd194cced2fc3fabd18754dd54218145929
```

## Usage

### Training with W&B

To enable W&B logging during training, add the `--use-wandb` flag:

```bash
python vim/main.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --batch-size 64 \
    --epochs 300 \
    --output_dir ./output/my_experiment \
    --use-wandb \
    --wandb-project visionmamba \
    --wandb-run-name my_experiment_run \
    --wandb-tags baseline zoh
```

**W&B Arguments:**
- `--use-wandb`: Enable W&B logging
- `--wandb-project`: Project name in W&B (default: "visionmamba")
- `--wandb-entity`: Your W&B username or team name (optional)
- `--wandb-run-name`: Custom run name (defaults to output_dir name)
- `--wandb-tags`: Space-separated tags for organizing runs (e.g., `--wandb-tags baseline zoh`)

### What Gets Logged

During training, W&B automatically logs:

**Training Metrics (per epoch):**
- `train/loss`: Training loss
- `train/lr`: Learning rate
- `train/img_s`: Training throughput (images/second)
- `train/samples/epoch`: Number of samples per epoch
- `train/epoch_time`: Time per epoch

**Validation Metrics (per epoch):**
- `val/loss`: Validation loss
- `val/acc1`: Top-1 accuracy
- `val/acc5`: Top-5 accuracy

**Training Metrics (per 100 iterations):**
- `train/iter_loss`: Iteration-level loss
- `train/iter_lr`: Iteration-level learning rate
- `train/iter_throughput`: Iteration-level throughput

**Other Metrics:**
- `epoch`: Current epoch number
- `max_accuracy`: Best validation accuracy so far
- `training/total_time_seconds`: Total training time

**Hyperparameters:**
All training arguments are automatically logged as config parameters.

### Benchmarking with W&B

To log benchmark results to W&B:

```bash
python performance-analysis/benchmark_latency_flops.py \
    --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
    --checkpoint ./output/classification_logs/vim_tiny_zoh/best_checkpoint.pth \
    --batch-size 1 \
    --output-dir ./output/classification_logs/vim_tiny_zoh/benchmark_results \
    --use-wandb \
    --wandb-project visionmamba-benchmarks \
    --wandb-run-name zoh_benchmark
```

**Benchmark Metrics Logged:**
- `benchmark/latency_avg_ms`: Average inference latency
- `benchmark/latency_min_ms`: Minimum latency
- `benchmark/latency_max_ms`: Maximum latency
- `benchmark/latency_median_ms`: Median latency
- `benchmark/throughput_img_per_sec`: Images per second
- `benchmark/flops_billion`: FLOPs in billions
- `model/total_params`: Total model parameters
- `model/trainable_params`: Trainable parameters
- `model/size_mb`: Model size in MB

## Viewing Results

1. **Web Dashboard**: After starting a run, W&B will print a URL like:
   ```
   https://wandb.ai/your-username/visionmamba/runs/abc123
   ```
   Open this URL in your browser to view real-time metrics.

2. **Compare Runs**: In the W&B dashboard, you can:
   - Compare multiple runs side-by-side
   - Filter by tags, hyperparameters, or metrics
   - Create custom plots and tables
   - Export results to CSV

3. **Command Line**: View runs from terminal:
   ```bash
   wandb status  # Show current run status
   wandb sync    # Sync offline runs
   ```

## Best Practices

1. **Use Descriptive Run Names**: Include key hyperparameters or experiment purpose:
   ```bash
   --wandb-run-name "vim_tiny_zoh_lr5e4_bs64"
   ```

2. **Use Tags for Organization**: Tag related experiments:
   ```bash
   --wandb-tags baseline ablation-study zoh
   ```

3. **Separate Projects**: Use different projects for different tasks:
   - `visionmamba` for training experiments
   - `visionmamba-benchmarks` for performance benchmarks
   - `visionmamba-inference` for inference analysis

4. **Monitor System Metrics**: W&B automatically tracks GPU/CPU usage and memory. Check the "System" tab in the dashboard.

5. **Save Important Checkpoints**: Use W&B artifacts to version model checkpoints:
   ```python
   # In your code (future enhancement)
   artifact = wandb.Artifact('model-checkpoint', type='model')
   artifact.add_file('checkpoint.pth')
   wandb.log_artifact(artifact)
   ```

## Troubleshooting

**Issue: "wandb: ERROR No API key found"**
- Solution: Run `wandb login` with your API key or set `WANDB_API_KEY` environment variable

**Issue: "wandb: ERROR Network error"**
- Solution: Check your internet connection. W&B requires network access to sync data.

**Issue: "wandb: ERROR Permission denied"**
- Solution: Make sure you're logged in with the correct account that has access to the project/entity.

**Issue: Metrics not appearing**
- Solution: Ensure `--use-wandb` flag is set and you're running on the main process (rank 0) in distributed training.

## Disabling W&B

If you don't want to use W&B, simply omit the `--use-wandb` flag. The code will work normally without W&B logging.

## Integration with Existing MLflow

This codebase also uses MLflow for logging. W&B and MLflow can run simultaneously - they serve complementary purposes:
- **MLflow**: Local experiment tracking and model registry
- **W&B**: Cloud-based visualization and collaboration

Both will log the same metrics, so you can use whichever platform you prefer or both together.

## Additional Resources

- [W&B Documentation](https://docs.wandb.ai/)
- [W&B Python API Reference](https://docs.wandb.ai/ref/python)
- [W&B Best Practices](https://docs.wandb.ai/guides/track)

