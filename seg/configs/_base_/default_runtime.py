# yapf:disable
# MMEngine ≥ 0.7: log_config is handled via default_hooks
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
# workflow is deprecated in MMEngine - use train_cfg/val_cfg instead
cudnn_benchmark = True
