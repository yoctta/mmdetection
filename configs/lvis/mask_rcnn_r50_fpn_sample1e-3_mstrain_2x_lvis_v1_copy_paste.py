_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/lvis_v1_instance.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
dataset_type = 'LVISV1Dataset'
data_root = '/mnt/data/LVIS/'
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1203), mask_head=dict(num_classes=1203)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='CopyPaste',max_num_pasted=5, bbox_occluded_thr=10, mask_occluded_thr=300, selected=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'lvis_v1_train.json',
            img_prefix=data_root,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Resize',
                    img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                            (1333, 768), (1333, 800)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=32)]
            )
    ),
    pipeline=train_pipeline)

data = dict(train=train_dataset)
evaluation = dict(classwise=True)