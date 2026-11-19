import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset
from mmengine.config import Config
from mmengine.runner import Runner

import dicom_pipeline  # registra LoadDicomImage
import mmdet.models  # registra FasterRCNN e todos os modelos
import mmdet.datasets  # registra todos os datasets
import mmdet.evaluation  # registra CocoMetric


@DATASETS.register_module(force=True)
class CbisDdsmDataset(CocoDataset):
    METAINFO = {
        'classes': ('mass',),
        'palette': [(220, 20, 60)]
    }


def main():
    # 1. Carregar config
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/cbis_faster_rcnn.py'
    cfg = Config.fromfile(config_path)

    # 2. Garantir que o diretório de trabalho existe
    os.makedirs(cfg.work_dir, exist_ok=True)

    print("=" * 50)
    print("Treinamento Digital Eye — CBIS-DDSM com DICOM")
    print("=" * 50)
    print(f"Config:    {config_path}")
    print(f"Modelo:    {cfg.model.type}")
    print(f"Épocas:    {cfg.train_cfg['max_epochs']}")
    print(f"Saída:     {cfg.work_dir}")
    print("=" * 50)

    # 3. Construir e iniciar o runner
    runner = Runner.from_cfg(cfg)
    runner.train()

    print("\nTreinamento concluído!")
    print(f"Checkpoints salvos em: {cfg.work_dir}")


if __name__ == '__main__':
    main()