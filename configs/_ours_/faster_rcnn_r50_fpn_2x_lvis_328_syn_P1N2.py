_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_ours_/lvis_v1_longtail_328.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py','./cat2label.py'
]

custom_imports = dict(imports=['configs._ours_.my_pipeline'], allow_failed_imports=False)


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=328)),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0001,
            # LVIS allows up to 300
            max_per_img=300)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(type='In_N_Out',P=1,N=2),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(train=dict(dataset=dict(pipeline=train_pipeline)))
evaluation = dict(classwise=True,remap_cat_id=cat2label)
##
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'