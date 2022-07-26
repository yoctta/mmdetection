# dataset settings
_base_ = '../_base_/datasets/coco_instance.py'
dataset_type = 'LVISV1Dataset'
data_root = '/mnt/data/LVIS'
import json
classes = json.load(open('/mnt/home/syn4det/GLIDE/LVIS_gen_FG/results.json'))
classes= [i['name'] for i in classes]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            classes=classes,
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_train.json',
            img_prefix=data_root)),
    val=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix=data_root))
evaluation = dict(metric=['bbox'])
