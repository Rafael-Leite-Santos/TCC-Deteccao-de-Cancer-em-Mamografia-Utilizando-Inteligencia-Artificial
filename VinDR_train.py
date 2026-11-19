"""
VinDR_train.py - treina e avalia o detector de massas no VinDr-Mammo

Uso (na raiz do projeto, com o venv ativo):
  python VinDR_train.py                         treina com configs/VinDR_faster_rcnn.py
                                                     e no fim avalia o melhor checkpoint
                                                     no split "test" oficial
  python VinDR_train.py configs/outro.py        treina com outro config
  python VinDR_train.py --retomar               continua um treino interrompido
                                                     (do ultimo checkpoint do work_dir)
  python VinDR_train.py --testar caminho.pth    so avalia um checkpoint no "test"

Antes do primeiro treino: python VinDR_prepare.py
"""
import argparse
import glob
import os
import time

from mmengine.config import Config
from mmengine.runner import Runner

import mmdet.models      # noqa: F401  registra FasterRCNN e demais modelos
import mmdet.datasets    # noqa: F401  registra os datasets
import mmdet.evaluation  # noqa: F401  registra o CocoMetric
import VinDR_pipeline    # noqa: F401  registra LoadVindrDicom e VindrMammoDataset

CONFIG_PADRAO = 'configs/VinDR_faster_rcnn.py'


def checar_anotacoes(cfg):
    faltando = [f for f in ('train.json', 'val.json', 'test.json')
                if not os.path.isfile(os.path.join(cfg.ANNOTS_DIR, f))]
    if faltando:
        raise SystemExit(f'Anotacoes nao encontradas em {cfg.ANNOTS_DIR}: {faltando}\n'
                         'Rode primeiro: python VinDR_prepare.py')
    if not os.path.isdir(cfg.IMAGES_ROOT):
        raise SystemExit(f'Pasta de imagens nao encontrada: {cfg.IMAGES_ROOT}\n'
                         'Ajuste VINDR_ROOT no config.')


def melhor_checkpoint(work_dir):
    candidatos = glob.glob(os.path.join(work_dir, 'best_*.pth'))
    return max(candidatos, key=os.path.getmtime) if candidatos else None


def treinar(config_path, retomar=False):
    cfg = Config.fromfile(config_path)
    checar_anotacoes(cfg)
    cfg.resume = retomar

    print('=' * 60)
    print(f'Config:    {config_path}')
    print(f'Modelo:    {cfg.model.type}')
    print(f'Imagens:   {cfg.IMAGES_ROOT}')
    print(f'Anotacoes: {cfg.ANNOTS_DIR}')
    print(f'Saida:     {cfg.work_dir}')
    print('=' * 60)

    runner = Runner.from_cfg(cfg)
    runner.train()
    return cfg.work_dir


def testar(config_path, checkpoint):
    """Avalia o checkpoint no split 'test' oficial do VinDr."""
    cfg = Config.fromfile(config_path)
    checar_anotacoes(cfg)
    cfg.load_from = checkpoint
    cfg.resume = False
    cfg.work_dir = os.path.join(cfg.work_dir, 'teste')
    cfg.experiment_name = 'teste_' + time.strftime('%Y%m%d_%H%M%S')

    print('=' * 60)
    print(f'Avaliando no split TEST: {checkpoint}')
    print('=' * 60)
    metricas = Runner.from_cfg(cfg).test()

    print('\nResultado no TEST (VinDr-Mammo):')
    for chave in ('coco/bbox_mAP_50', 'coco/bbox_mAP', 'coco/bbox_mAP_75'):
        if chave in metricas:
            print(f'  {chave:20s} {metricas[chave] * 100:.1f}%')
    return metricas


def main():
    p = argparse.ArgumentParser(description='Treino do detector de massas no VinDr-Mammo')
    p.add_argument('config', nargs='?', default=CONFIG_PADRAO)
    p.add_argument('--retomar', action='store_true',
                   help='continua do ultimo checkpoint do work_dir')
    p.add_argument('--testar', metavar='CHECKPOINT',
                   help='so avalia este checkpoint no split test (nao treina)')
    p.add_argument('--sem-teste', action='store_true',
                   help='nao avalia no split test ao fim do treino')
    args = p.parse_args()

    if args.testar:
        testar(args.config, args.testar)
        return

    work_dir = treinar(args.config, args.retomar)

    if not args.sem_teste:
        melhor = melhor_checkpoint(work_dir)
        if melhor:
            testar(args.config, melhor)
        else:
            print('Nenhum best_*.pth encontrado; avaliacao no test pulada.')


if __name__ == '__main__':   # obrigatorio no Windows com num_workers > 0
    main()
