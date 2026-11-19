# ============================================================
# CONFIG: Cascade R-CNN + ResNet-50 para CBIS-DDSM (mmdet 3.x)
# ------------------------------------------------------------
# Mesmo dataset, pipeline DICOM, RPN e agendamento do treino
# Faster R-CNN (cbis_faster_rcnn.py). A unica mudanca e a
# cabeca RoI: 3 estagios em cascata, com IoU 0,5 -> 0,6 -> 0,7,
# que refinam a caixa a cada estagio.
# ============================================================

default_scope = 'mmdet'

custom_imports = dict(
    imports=['dicom_pipeline'],
    allow_failed_imports=False
)

# ---- MODELO ----
model = dict(
    type='CascadeRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    # RPN identica a do Faster R-CNN (mesmas ancoras e losses)
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True,
                      loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    # Cabeca em cascata: o que muda em relacao ao Faster R-CNN
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            # estagio 1 (IoU 0,5)
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False,
                              loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            # estagio 2 (IoU 0,6)
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False,
                              loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            ),
            # estagio 3 (IoU 0,7)
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]
                ),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False,
                              loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
            )
        ]
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7,
                          neg_iou_thr=0.3, min_pos_iou=0.3,
                          match_low_quality=True, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5,
                         neg_pos_ub=-1, add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000, max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        # um assigner por estagio, com IoU crescente
        rcnn=[
            dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5,
                              neg_iou_thr=0.5, min_pos_iou=0.5,
                              match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False
            ),
            dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.6,
                              neg_iou_thr=0.6, min_pos_iou=0.6,
                              match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False
            ),
            dict(
                assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7,
                              neg_iou_thr=0.7, min_pos_iou=0.7,
                              match_low_quality=False, ignore_iof_thr=-1),
                sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                             neg_pos_ub=-1, add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False
            )
        ]
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05,
                  nms=dict(type='nms', iou_threshold=0.5),
                  max_per_img=100)
    )
)

# ---- DATASET (igual ao Faster R-CNN) ----
IMAGES_ROOT = 'data/CBIS-DDSM/imagens_completas'   # ajuste para a sua maquina
ANNOTS_DIR  = 'annotations'                        # saida do prepare_cbis_final.py

train_pipeline = [
    dict(type='LoadDicomImage'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

val_pipeline = [
    dict(type='LoadDicomImage'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=2,       # reduza para 1 se der erro de memoria
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CbisDdsmDataset',
        data_root='',                      # vazio: ann_file fica relativo a raiz do clone
        ann_file=ANNOTS_DIR + '/train.json',
        data_prefix=dict(img=IMAGES_ROOT),  # pasta das imagens
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CbisDdsmDataset',
        data_root='',                      # vazio: ann_file fica relativo a raiz do clone
        ann_file=ANNOTS_DIR + '/val.json',
        data_prefix=dict(img=IMAGES_ROOT),  # pasta das imagens
        pipeline=val_pipeline
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=ANNOTS_DIR + '/val.json',
    metric='bbox',
    format_only=False
)

# ---- TREINO (igual ao ultimo Faster R-CNN) ----
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg   = dict(type='ValLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='CosineAnnealingLR', T_max=40, eta_min=1e-5,
         begin=0, end=40, by_epoch=True, convert_to_iter_based=True)
]

# ---- LOG, CHECKPOINT E PARADA (pelo mAP@0,50) ----
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50',
        rule='greater'
    ),
    logger=dict(type='LoggerHook', interval=50)
)

custom_hooks = [
    dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP_50',
        rule='greater',
        min_delta=0.005,
        patience=8
    )
]

work_dir = './work_dirs/cbis_cascade_rcnn_40ep_parada_map50'

# Pesos iniciais: so o backbone pre-treinado no ImageNet, como no Faster R-CNN.
# Para um teste posterior com pesos do COCO, coloque aqui o link do checkpoint
# "cascade-rcnn_r50_fpn_1x_coco" da pagina configs/cascade_rcnn do MMDetection.
load_from = None

# ---- NECESSARIO PARA INFERENCIA ----
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CbisDdsmDataset',
        data_root='',                      # vazio: ann_file fica relativo a raiz do clone
        ann_file=ANNOTS_DIR + '/val.json',
        data_prefix=dict(img=IMAGES_ROOT),  # pasta das imagens
        pipeline=val_pipeline
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=ANNOTS_DIR + '/val.json',
    metric='bbox',
    format_only=False
)

test_cfg = dict(type='TestLoop')