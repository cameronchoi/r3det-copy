_base_ = [
    '../_base_/models/r3det_r50_fpn.py',
    '../_base_/datasets/dotav1_rotational_detection.py',
    '../_base_/schedules/schedule_1x.py'
]

# runtime settings
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './work_dirs/r3det_r50_fpn_2x_20200616'
evaluation = dict(interval=1, metric='mAP')
