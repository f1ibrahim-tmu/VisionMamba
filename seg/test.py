# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import mmengine
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmengine.dist import get_dist_info, init_dist
from mmengine.runner import load_checkpoint, wrap_fp16_model
from mmengine.config import DictAction

# MMSegmentation 1.0.0+ moved digit_version to mmengine
try:
    from mmengine.utils import digit_version
except ImportError:
    # Fallback for older versions
    from mmseg import digit_version
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
from mmengine.runner import Runner
from mmengine.model import MMDataParallel, MMDistributedDataParallel

from backbone import vim


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmengine.Config.fromfile(args.config)
    print("cfg: ", cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmengine.utils.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmengine.utils.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataset and dataloader
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    test_dataloader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Handle fp16 - use wrap_fp16_model for inference (deprecated but still works)
    # Note: For training, use AmpOptimWrapper instead
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'
    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmengine.utils.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    # Prepare evaluator configuration
    # MMEngine uses evaluator instead of direct evaluation
    test_evaluator = None
    if args.eval is not None or args.format_only:
        # Build evaluator from mmseg
        try:
            from mmseg.engine import SegEvaluator
            evaluator_cfg = dict(
                type='IoUMetric' if args.eval else 'SegEvaluator',
                format_only=args.format_only or eval_on_format_results
            )
            if args.eval:
                evaluator_cfg['metric'] = args.eval
            evaluator_cfg.update(eval_kwargs)
            test_evaluator = SegEvaluator(**evaluator_cfg) if hasattr(SegEvaluator, '__call__') else evaluator_cfg
        except (ImportError, TypeError):
            # Fallback: use dict config for evaluator
            evaluator_cfg = dict(
                type='IoUMetric' if args.eval else 'SegEvaluator',
                format_only=args.format_only or eval_on_format_results
            )
            if args.eval:
                evaluator_cfg['metric'] = args.eval
            evaluator_cfg.update(eval_kwargs)
            test_evaluator = evaluator_cfg

    # Set up device and model wrapping
    cfg.device = get_device()
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            # MMCV 2.x is compatible with CPU training
            import mmengine
            assert digit_version(mmengine.__version__) >= digit_version('0.10.0'), \
                'Please use MMEngine >= 0.10.0 for CPU training!'
        model = revert_sync_batchnorm(model)
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    else:
        model = MMDistributedDataParallel(
            model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)

    # Create Runner for testing
    # Note: For custom show/format functionality, we may need to use custom hooks
    # For now, we'll use Runner.test() and handle show/format separately if needed
    test_cfg = dict(type='TestLoop')
    
    # Set up work_dir
    if args.work_dir is not None:
        work_dir = args.work_dir
    else:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
    
    runner = Runner(
        model=model,
        work_dir=work_dir,
        test_dataloader=test_dataloader,
        test_evaluator=test_evaluator,
        test_cfg=test_cfg,
        default_scope='mmseg'
    )

    # Register custom hooks for show/format/out functionality if needed
    custom_hooks = []
    if args.show or args.show_dir:
        # Note: Show functionality typically requires custom visualization hooks
        # This is a placeholder - full implementation would need a custom hook
        warnings.warn(
            'Show functionality with Runner.test() requires custom hooks. '
            'Consider using mmseg visualization hooks.')
    
    if custom_hooks:
        runner.register_hook(custom_hooks[0])
    
    # Run testing
    # Runner.test() handles distributed testing automatically
    runner.test()

    # Handle post-test outputs
    rank, _ = get_dist_info()
    if rank == 0:
        # Get results from runner's message hub or evaluator
        # The evaluator should have stored results during test()
        results = None
        if hasattr(runner, 'message_hub'):
            # Try to get results from message hub
            if 'test' in runner.message_hub.log_scalars:
                results = runner.message_hub.log_scalars.get('test', {})
        
        # Save results to file if requested
        if args.out:
            if results is None:
                warnings.warn(
                    'Results not available from Runner. '
                    'The --out option may not work as expected with Runner.test(). '
                    'Consider using evaluator outputs or custom hooks.')
            else:
                print(f'\nwriting results to {args.out}')
                mmengine.fileio.dump(results, args.out)
        
        # Save evaluation metrics if evaluation was performed
        if args.eval and test_evaluator is not None:
            # Evaluation results should be in runner's message hub
            # or printed by the evaluator
            if hasattr(runner, 'message_hub'):
                eval_results = runner.message_hub.log_scalars.get('test', {})
                if eval_results:
                    metric_dict = dict(config=args.config, metric=eval_results)
                    mmengine.fileio.dump(metric_dict, json_file, indent=4)
        
        # Clean up tmpdir if created
        if tmpdir is not None and eval_on_format_results:
            shutil.rmtree(tmpdir)


if __name__ == '__main__':
    main()