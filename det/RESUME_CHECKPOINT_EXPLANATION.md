# Resume and Checkpoint Explanation

## Question 1: Where does the detection code support resume via `--resume` flag?

### Answer: The `--resume` flag is supported in multiple places:

#### 1. **Argument Parser Definition**
Location: `det/detectron2/engine/defaults.py:110-115`

```python
parser.add_argument(
    "--resume",
    action="store_true",
    help="Whether to attempt to resume from the checkpoint directory. "
    "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
)
```

#### 2. **Training Script Usage**
Location: `det/tools/lazyconfig_train_net.py:108-115`

```python
checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
if args.resume and checkpointer.has_checkpoint():
    # The checkpoint stores the training iteration that just finished, thus we start
    # at the next iteration
    start_iter = trainer.iter + 1
else:
    start_iter = 0
trainer.train(start_iter, cfg.train.max_iter)
```

**Key behavior:**
- When `args.resume=True`, the checkpointer looks for a `last_checkpoint` file in `train.output_dir`
- If found, it loads the **full training state** (model + optimizer + scheduler + iteration count)
- Training continues from `trainer.iter + 1`

#### 3. **Resume Logic Implementation**
Location: `det/detectron2/engine/defaults.py:406-424`

```python
def resume_or_load(self, resume=True):
    """
    If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
    a `last_checkpoint` file), resume from the file. Resuming means loading all
    available states (eg. optimizer and scheduler) and update iteration counter
    from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.

    Otherwise, this is considered as an independent training. The method will load model
    weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
    from iteration 0.
    """
    self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
    if resume and self.checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        self.start_iter = self.iter + 1
```

**Important points:**
- When `resume=True`, it **ignores** `cfg.MODEL.WEIGHTS` (or `cfg.train.init_checkpoint`)
- It automatically finds the last checkpoint via the `last_checkpoint` file in the output directory
- Loads **everything**: model weights, optimizer state, scheduler state, iteration number

---

## Question 2: Do you need `train.init_checkpoint`? What is it for if you have `PRETRAIN_CKPT`?

### Answer: **Yes, you need `train.init_checkpoint`, but for a different purpose than `PRETRAIN_CKPT`**

#### **`PRETRAIN_CKPT` (via `model.backbone.net.pretrained`)**
- **Purpose**: Loads **pretrained backbone weights** during model initialization
- **When**: Used when building the model, before training starts
- **What it loads**: Only the backbone weights (Vision Mamba classification weights)
- **Location**: `det/detectron2/modeling/backbone/vim.py:82-126`
- **Usage**: `model.backbone.net.pretrained=${PRETRAIN_CKPT}` in your config

```python
def init_weights(self, pretrained=None):
    """Initialize the weights in backbone."""
    if isinstance(pretrained, str) and pretrained:
        state_dict = torch.load(pretrained, map_location="cpu")
        state_dict_model = state_dict["model"]
        state_dict_model.pop("head.weight")  # Remove classification head
        state_dict_model.pop("head.bias")
        # Load only backbone weights
```

**This is for**: Initializing your detection model's backbone with pretrained Vision Mamba weights from classification training.

#### **`train.init_checkpoint`**
- **Purpose**: Loads a **full detection model checkpoint** (entire model, not just backbone)
- **When**: Used when **NOT resuming** (i.e., when `--resume` is False)
- **What it loads**: The entire detection model (backbone + detection heads + optimizer + scheduler if present)
- **Location**: `det/tools/lazyconfig_train_net.py:108`

```python
checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
```

**This is for**: 
1. **Starting from a previous detection checkpoint** (if you want to continue from a specific checkpoint, not the last one)
2. **Evaluation** - loading a trained detection model for inference
3. **Transfer learning** - loading a detection model trained on a different dataset

#### **Key Differences:**

| Feature | `PRETRAIN_CKPT` (pretrained) | `train.init_checkpoint` |
|---------|------------------------------|-------------------------|
| **Scope** | Backbone only | Full detection model |
| **When used** | Model initialization | Training start (if not resuming) |
| **Contains** | Classification weights | Detection weights + heads |
| **Purpose** | Initialize backbone | Load full detection checkpoint |
| **Ignored when** | Never | When `--resume=True` |

#### **In Your Case:**

You're setting `train.init_checkpoint=""` (empty string), which means:
- **First training run**: No detection checkpoint to load, so it starts fresh
  - Backbone is initialized from `PRETRAIN_CKPT` (classification weights)
  - Detection heads start from random initialization
- **Resume (with `--resume`)**: The checkpointer finds the last checkpoint automatically
  - `train.init_checkpoint=""` is **ignored** when `--resume=True`
  - Full training state is restored from the last checkpoint

#### **When You Would Use `train.init_checkpoint`:**

1. **Loading a specific checkpoint** (not the last one):
   ```bash
   train.init_checkpoint="output/detection_logs/vim_tiny_vimdet_rk4/checkpoint_0050000.pth"
   ```

2. **Evaluation** (as seen in `det/scripts/eval_vim_tiny_vimdet.sh`):
   ```bash
   --eval-only train.init_checkpoint="/path/to/trained/model.pth"
   ```

3. **Transfer learning** from a different detection model:
   ```bash
   train.init_checkpoint="/path/to/other/detection/model.pth"
   ```

---

## Summary

1. **`--resume` flag**: Fully supported in your detection code. It automatically finds and loads the last checkpoint from `train.output_dir/last_checkpoint`.

2. **`train.init_checkpoint`**: You need it, but setting it to `""` is correct for your use case. It's only used when NOT resuming. When `--resume=True`, it's ignored.

3. **`PRETRAIN_CKPT`**: Different purpose - initializes backbone weights from classification training, used during model building.

4. **Your setup is correct**: 
   - `train.init_checkpoint=""` → Start fresh or let resume handle it
   - `model.backbone.net.pretrained=${PRETRAIN_CKPT}` → Initialize backbone from classification weights
   - `--resume` flag → Automatically resume from last checkpoint if it exists
