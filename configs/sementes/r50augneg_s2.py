# ============================================================
# CONFIG: Faster R-CNN + ResNet-50 para VinDr-Mammo (mmdet 3.x)
# ------------------------------------------------------------
# Mesmo modelo e agendamento do ultimo Faster R-CNN do CBIS-DDSM
# (40 epocas, cosseno, parada pelo mAP@0,50 com paciencia 8).
# VARIANTE AUG (17/11/2026, ponto 3 do orientador): igual ao VinDR_faster_rcnn_rot.py
# (flip + rotacao +-10 graus) e acrescenta escala (+-20%), translacao (ate 10%),
# brilho e contraste (fator 0,8-1,2), cada um com probabilidade 0,5. So no treino.
# Muda so o dataset e a leitura do DICOM (LoadVindrDicom), para
# que a comparacao CBIS x VinDr meca o efeito dos dados.
# ============================================================

default_scope = 'mmdet'

# NEGATIVAS (23/11/2026): copia de configs/sementes/r50aug_s2.py. Muda so o TREINO: imagens com massa
# + 1 imagem sem achado por imagem com massa (annotations_VinDR_neg1, gerado pelo VinDR_prepare_neg.py).
# Validacao e teste continuam os de annotations_VinDR (so com massa), para comparar com os demais treinos.
# SEMENTE FIXA (17/11/2026): copia de configs/VinDR_faster_rcnn_aug.py, mudando so a semente e o work_dir.
randomness = dict(seed=2, deterministic=False)

custom_imports = dict(
    imports=['VinDR_pipeline'],      # registra LoadVindrDicom e VindrMammoDataset
    allow_failed_imports=False
)

# ---- CAMINHOS (ajuste aqui) ----
VINDR_ROOT  = 'data/VinDr-Mammo'                 # pasta do dataset
IMAGES_ROOT = VINDR_ROOT + '/images'                           # images/<study_id>/<image_id>.dicom
ANNOTS_DIR  = 'annotations_VinDR'  # saida do VinDR_prepare.py
ANNOTS_TRAIN = 'annotations_VinDR_neg1'  # treino com negativas (VinDR_prepare_neg.py)

# ---- MODELO ----
model = dict(
    type='FasterRCNN',
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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[4, 8, 16],          # ancoras menores, para massas pequenas
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1,              # so 'mass'
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3,
                          min_pos_iou=0.3, match_low_quality=True, ignore_iof_thr=-1),
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
        rcnn=dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5,
                          min_pos_iou=0.5, match_low_quality=False, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25,
                         neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(nms_pre=1000, max_per_img=1000,
                 nms=dict(type='nms', iou_threshold=0.7), min_bbox_size=0),
        rcnn=dict(score_thr=0.05,
                  nms=dict(type='nms', iou_threshold=0.5),
                  max_per_img=100)
    )
)

# ---- PIPELINES ----
train_pipeline = [
    dict(type='LoadVindrDicom'),
    dict(type='LoadAnnotations', with_bbox=True),
    # Escala: o lado alvo sorteado entre 0,8x e 1,2x de 1024 (validacao/teste seguem em 1024)
    dict(type='RandomResize', scale=(1024, 1024), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    # Rotacao: 50% das vezes, ate 10 graus para um lado ou outro; borda preta (0)
    dict(type='Rotate', prob=0.5, max_mag=10.0, reversal_prob=0.5,
         img_border_value=0),
    # Translacao: ate 10% da largura/altura, para um lado ou outro; borda preta
    dict(type='TranslateX', prob=0.5, max_mag=0.1, reversal_prob=0.5, img_border_value=0),
    dict(type='TranslateY', prob=0.5, max_mag=0.1, reversal_prob=0.5, img_border_value=0),
    # Brilho e contraste: fator sorteado entre 0,8 e 1,2 (1,0 = sem mudanca)
    dict(type='Brightness', prob=0.5, min_mag=0.8, max_mag=1.2),
    dict(type='Contrast', prob=0.5, min_mag=0.8, max_mag=1.2),
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
        data_root='',                      # vazio: ann_file fica relativo a raiz do clone
        ann_file=ANNOTS_TRAIN + '/train.json',
        data_prefix=dict(img=IMAGES_ROOT),  # imagens: images/<study_id>/<image_id>.dicom
        filter_cfg=dict(filter_empty_gt=False),  # mantem as imagens sem massa no treino
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
        data_root='',                      # vazio: ann_file fica relativo a raiz do clone
        ann_file=ANNOTS_DIR + '/val.json',
        data_prefix=dict(img=IMAGES_ROOT),  # imagens: images/<study_id>/<image_id>.dicom
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
        data_root='',                      # vazio: ann_file fica relativo a raiz do clone
        ann_file=ANNOTS_DIR + '/test.json',
        data_prefix=dict(img=IMAGES_ROOT),  # imagens: images/<study_id>/<image_id>.dicom
        test_mode=True,
        pipeline=val_pipeline
    )
)

val_evaluator = dict(type='CocoMetric', ann_file=ANNOTS_DIR + '/val.json',
                     metric='bbox', format_only=False)
test_evaluator = dict(type='CocoMetric', ann_file=ANNOTS_DIR + '/test.json',
                      metric='bbox', format_only=False)

# ---- TREINO (igual ao ultimo Faster R-CNN do CBIS) ----
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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

work_dir = './work_dirs/sementes/r50augneg_s2'
load_from = None   # so o backbone do ImageNet, como no CBIS
