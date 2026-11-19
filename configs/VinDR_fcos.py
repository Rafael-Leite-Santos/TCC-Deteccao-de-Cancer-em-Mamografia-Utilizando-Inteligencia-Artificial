# ============================================================
# CONFIG: FCOS + ResNet-50 para VinDr-Mammo (mmdet 3.x)
# ------------------------------------------------------------
# Segundo/terceiro modelo do ensemble (Faster R-CNN + FCOS + ATSS).
# Dados, pipeline (LoadVindrDicom, sem recorte, Resize 1024), divisao,
# agendamento (40 epocas, cosseno) e parada pelo mAP@0,50 (paciencia 8)
# IGUAIS aos do configs/VinDR_faster_rcnn.py: muda so o detector.
# ============================================================

default_scope = 'mmdet'

custom_imports = dict(
    imports=['VinDR_pipeline'],      # registra LoadVindrDicom e VindrMammoDataset
    allow_failed_imports=False
)

# ---- CAMINHOS (ajuste aqui) ----
VINDR_ROOT  = 'data/VinDr-Mammo'                 # pasta do dataset
IMAGES_ROOT = VINDR_ROOT + '/images'                           # images/<study_id>/<image_id>.dicom
ANNOTS_DIR  = 'annotations_VinDR'  # saida do VinDR_prepare.py

# ---- MODELO ----
# FCOS (Tian et al., 2019): detector de um estagio SEM ancoras. Cada posicao dos
# mapas P3-P7 que cai dentro de uma massa preve a distancia ate as 4 bordas da
# caixa, a classe e a "centerness" (quao perto do centro da massa ela esta).
model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],     # mesmas estatisticas do ImageNet do Faster R-CNN
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    # Mesmo backbone do Faster R-CNN (ResNet-50 do torchvision, pre-treinada no
    # ImageNet), para que a diferenca entre os modelos venha so da cabeca
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,                   # P3 a P7 (passos 8 a 128)
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True
    ),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=1,                   # so 'mass'
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),   # GroupNorm na cabeca
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100
    )
)

# ---- PIPELINES ----
train_pipeline = [
    dict(type='LoadVindrDicom'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# Resize ANTES das anotacoes: as caixas ficam na escala original e o
# CocoMetric compara na escala original (igual ao CBIS)
val_pipeline = [
    dict(type='LoadVindrDicom'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# ---- DATALOADERS ----
train_dataloader = dict(
    batch_size=2,           # reduza para 1 se faltar memoria na GPU
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='VindrMammoDataset',
        data_root=IMAGES_ROOT,
        ann_file=ANNOTS_DIR + '/train.json',
        data_prefix=dict(img=''),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VindrMammoDataset',
        data_root=IMAGES_ROOT,
        ann_file=ANNOTS_DIR + '/val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline
    )
)

# Teste = split "test" oficial do VinDr (so usado no fim do treino)
test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VindrMammoDataset',
        data_root=IMAGES_ROOT,
        ann_file=ANNOTS_DIR + '/test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=val_pipeline
    )
)

val_evaluator = dict(type='CocoMetric', ann_file=ANNOTS_DIR + '/val.json',
                     metric='bbox', format_only=False)
test_evaluator = dict(type='CocoMetric', ann_file=ANNOTS_DIR + '/test.json',
                      metric='bbox', format_only=False)

# ---- TREINO (igual ao Faster R-CNN do VinDr) ----
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Taxa: padrao do FCOS no MMDetection (0,01 para lote de 16) escalada para o
# lote de 2, como foi feito com o Faster R-CNN (0,02 -> 0,0025). Bias com taxa
# 2x e sem decaimento, e corte do gradiente: padrao do FCOS (e do Digital Eye).
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.),
    clip_grad=dict(max_norm=35, norm_type=2)
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

work_dir = './work_dirs/VinDR_fcos_40ep_parada_map50'
load_from = None   # so o backbone do ImageNet, como no Faster R-CNN
