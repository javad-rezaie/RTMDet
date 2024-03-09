_base_ = [
    'mmdet::rtmdet/rtmdet_s_8xb32-300e_coco.py'
]

# dataset settings
dataset_type = 'CocoDataset'
data_root = "/data/" 
train_annot = "train_coco.json"
val_annot = "test_coco.json"
test_annot = "test_coco.json"
train_image_folder = "images/"
val_image_folder = "images/"
test_image_folder = "images/"

# Training Parameter Settings
base_lr = 0.004
max_epochs = 100
warmup_iters = 200
check_point_interval = 10
val_interval =  1
stage2_num_epochs = 40

work_dir = "/out"

train_data_annot_path = data_root + train_annot

def get_object_classes(path_to_annotation):
    import json
    with open(path_to_annotation, "r") as f:
        data = json.load(f)
    cats = [cat['name'] for cat in data["categories"]]
    return tuple(cats)


classes = get_object_classes(train_data_annot_path)

metainfo = {
    'classes': classes
}
num_classes = len(classes)


model = dict(
    bbox_head=dict(
        num_classes = num_classes
    )
)


train_dataloader = dict(
    batch_size=8,
    num_workers=3,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=train_annot,
        data_prefix=dict(img=train_image_folder)
        )
    )

val_dataloader = dict(
    batch_size=8,
    num_workers=3,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=val_annot,
        data_prefix=dict(img=val_image_folder)
        )
    )

test_dataloader = dict(
    batch_size=8,
    num_workers=3,
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file=test_annot,
        data_prefix=dict(img=test_image_folder),
        test_mode=True
        )
    )

train_cfg = dict(
    val_interval=val_interval
    )


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_annot,
    metric='bbox')
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + test_annot,
    metric='bbox')


# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=warmup_iters),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(
        interval=check_point_interval, # Save checkpoint on "interval" epochs
        max_keep_ckpts=1  # only keep latest 1 checkpoints
    ))
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs, 
        switch_pipeline=_base_.train_pipeline_stage2
        )
]
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
# auto_scale_lr = dict(enable=True, base_batch_size=8*32)