"""
VinDR_ensemble.py - combina Faster R-CNN, FCOS e ATSS com Weighted Boxes Fusion

Mesma estrategia do Digital Eye (Terzi et al., 2025): as caixas dos modelos sao
fundidas com Weighted Boxes Fusion (Solovyev et al., 2021). Diferente do NMS,
que escolhe uma caixa e descarta as outras, o WBF agrupa as caixas que se
sobrepoem e faz a media das coordenadas, ponderada pela confianca.

O script:
  1. roda os 3 modelos em todas as imagens do conjunto (teste oficial por padrao),
     com o mesmo pre-processamento do treino (carregar_vindr), e guarda as
     deteccoes em cache (.json), para refazer a fusao sem rodar os modelos de novo;
  2. funde as deteccoes com WBF;
  3. avalia cada modelo sozinho e o ensemble com as metricas do COCO
     (mAP, mAP@0,50, mAP@0,75) e com TPR e FPPI (IoU >= 0,5, score >= --thr).

Uso:
  python VinDR_ensemble.py                       teste oficial, pesos iguais
  python VinDR_ensemble.py --split val           validacao (para escolher parametros)
  python VinDR_ensemble.py --pesos 1 1 2         peso maior para o 3o modelo
  python VinDR_ensemble.py --iou-wbf 0.5         limiar de IoU do WBF
  python VinDR_ensemble.py --refazer             ignora o cache e roda os modelos de novo

Os checkpoints sao os best_*.pth do work_dir de cada config. Se os work_dirs
estiverem em outra pasta (ex.: o clone do repositorio, sem os pesos), defina
WORK_DIRS com a pasta que contem as subpastas de cada modelo.

A interface web (main.py com ENSEMBLE=1) usa carregar_modelos() e
detectar_ensemble(), com os mesmos parametros padrao do WBF deste script.

Os parametros do ensemble (pesos, --iou-wbf) devem ser escolhidos na VALIDACAO;
o teste oficial so deve ser usado para reportar o resultado final.
"""
import argparse
import contextlib
import glob
import io
import json
import os
import time

import numpy as np
import torch
from ensemble_boxes import weighted_boxes_fusion
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from VinDR_pipeline import carregar_vindr  # tambem registra LoadVindrDicom e VindrMammoDataset
from metricas import THR_PADRAO, tpr_fppi

MODELOS = [
    # 18/11/2026: Faster R-CNN com o pacote de aumento de dados (semente 2, ep. 25),
    # escolhido pela validacao entre as 3 sementes. Antes: configs/VinDR_faster_rcnn.py.
    ('Faster R-CNN', 'configs/sementes/r50aug_s2.py'),
    # 19/11/2026: FCOS e ATSS com o mesmo pacote de aumento de dados, semente com o maior
    # mAP@0,50 de validacao entre 3 (FCOS s0: 79,3%; ATSS s0: 76,2%).
    # Antes: configs/VinDR_fcos.py e configs/VinDR_atss.py (sem aumento).
    ('FCOS',         'configs/sementes/fcosaug_s0.py'),
    ('ATSS',         'configs/sementes/atssaug_s0.py'),
]
PASTA_SAIDA = 'resultados_ensemble'
PESOS_PADRAO = [1, 1, 1]   # confirmado na validacao (melhor que [1, 0.5, 1])
IOU_WBF_PADRAO = 0.55
SKIP_PADRAO = 0.05


def melhor_checkpoint(work_dir):
    candidatos = glob.glob(os.path.join(work_dir, 'best_*.pth'))
    if not candidatos:
        raise SystemExit(f'Nenhum best_*.pth em {work_dir}: treine o modelo antes.')
    return max(candidatos, key=os.path.getmtime)


def pasta_do_modelo(config, work_dirs=None):
    """work_dir do config; com work_dirs (ou a variavel WORK_DIRS), a subpasta de
    mesmo nome dentro dessa pasta."""
    work_dir = Config.fromfile(config).work_dir
    work_dirs = work_dirs or os.environ.get('WORK_DIRS')
    if work_dirs:
        # mantem as subpastas abaixo de work_dirs (ex.: sementes/r50aug_s2), nao so o ultimo nome
        partes = os.path.normpath(work_dir).replace('\\', '/').split('/')
        rel = partes[partes.index('work_dirs') + 1:] if 'work_dirs' in partes else partes[-1:]
        return os.path.join(work_dirs, *rel)
    return work_dir


def carregar_modelos(device, work_dirs=None):
    """Os 3 modelos do ensemble: [(nome, config, checkpoint, modelo), ...]."""
    modelos = []
    for nome, config in MODELOS:
        ckpt = melhor_checkpoint(pasta_do_modelo(config, work_dirs))
        print(f'Carregando {nome}: {ckpt}')
        modelos.append((nome, config, ckpt, init_detector(config, ckpt, device=device)))
    return modelos


def detectar_ensemble(modelos, caminho, thr, pesos=None, iou_wbf=IOU_WBF_PADRAO,
                      skip=SKIP_PADRAO, filtros=True, so=None):
    """Uma imagem: roda os 3 modelos, funde com o mesmo fundir() da avaliacao e
    aplica os mesmos filtros do VinDR_inference.detectar(). Devolve
    (imagem, deteccoes, descartadas), no formato do detectar().
    so='<nome do modelo>' (ex.: 'FCOS'): roda so esse modelo, sem fusao (mesmos filtros);
    usado pelo seletor de modelo da interface (19/11/2026)."""
    from VinDR_inference import filtrar  # import aqui: so a interface precisa dele
    img, mascara = carregar_vindr(caminho)
    h, w = img.shape[:2]
    if so is not None:
        model = {nome: m for nome, _, _, m in modelos}[so]   # KeyError se o nome nao existir
        pred = inference_detector(model, img).pred_instances
        caixas, descartadas = filtrar(pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy(),
                                      mascara, thr, filtros)
        return img, caixas, descartadas
    deteccoes = {}
    for nome, _, _, model in modelos:
        pred = inference_detector(model, img).pred_instances
        deteccoes[nome] = {0: [[float(v) for v in b] + [float(s)] for b, s in
                               zip(pred.bboxes.cpu().numpy(), pred.scores.cpu().numpy())]}
    fundidas = fundir(deteccoes, [dict(id=0, width=w, height=h)],
                      pesos or PESOS_PADRAO, iou_wbf, skip)[0]
    d = np.array(fundidas, dtype=np.float64).reshape(-1, 5)
    caixas, descartadas = filtrar(d[:, :4], d[:, 4], mascara, thr, filtros)
    return img, caixas, descartadas


def rodar_modelos(cfg_base, split, refazer):
    """Deteccoes de cada modelo em cada imagem: {nome: {image_id: [[x1,y1,x2,y2,score], ...]}}."""
    cache = os.path.join(PASTA_SAIDA, f'deteccoes_{split}.json')
    ckpts = {nome: melhor_checkpoint(pasta_do_modelo(config)) for nome, config in MODELOS}
    if os.path.isfile(cache) and not refazer:
        with open(cache, encoding='utf-8') as f:
            dados = json.load(f)
        # o cache guarda quais checkpoints o geraram; se mudou algum modelo, roda de novo
        if dados.get('_checkpoints') == {n: os.path.normpath(c) for n, c in ckpts.items()}:
            print('Usando deteccoes em cache:', cache, '(use --refazer para rodar os modelos de novo)')
            return {n: {int(k): v for k, v in d.items()} for n, d in dados.items() if n != '_checkpoints'}
        print('Cache de outros checkpoints (ou sem registro): rodando os modelos de novo.')

    with open(os.path.join(cfg_base.ANNOTS_DIR, f'{split}.json'), encoding='utf-8') as f:
        imagens = json.load(f)['images']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    modelos = [(nome, model) for nome, _, _, model in carregar_modelos(device)]

    deteccoes = {nome: {} for nome, _ in modelos}
    t0 = time.time()
    for i, im in enumerate(imagens, 1):
        img, _ = carregar_vindr(os.path.join(cfg_base.IMAGES_ROOT, im['file_name']))  # le o DICOM uma vez
        for nome, model in modelos:
            pred = inference_detector(model, img).pred_instances
            caixas = pred.bboxes.cpu().numpy()
            scores = pred.scores.cpu().numpy()
            deteccoes[nome][im['id']] = [[float(v) for v in b] + [float(s)] for b, s in zip(caixas, scores)]
        if i % 20 == 0 or i == len(imagens):
            print(f'  {i}/{len(imagens)} imagens ({time.time() - t0:.0f} s)')

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    with open(cache, 'w', encoding='utf-8') as f:
        json.dump(dict(deteccoes, _checkpoints={n: os.path.normpath(c) for n, c in ckpts.items()}), f)
    return deteccoes


def fundir(deteccoes, imagens, pesos, iou_wbf, skip):
    """Weighted Boxes Fusion por imagem. As caixas sao normalizadas para 0-1, como exige o WBF."""
    nomes = [n for n, _ in MODELOS]
    resultado = {}
    for im in imagens:
        w, h = im['width'], im['height']
        escala = np.array([w, h, w, h], dtype=np.float64)
        caixas, scores, labels = [], [], []
        for nome in nomes:
            d = np.array(deteccoes[nome].get(im['id'], []), dtype=np.float64).reshape(-1, 5)
            caixas.append(np.clip(d[:, :4] / escala, 0, 1).tolist())
            scores.append(d[:, 4].tolist())
            labels.append([0] * len(d))
        if sum(len(s) for s in scores) == 0:
            resultado[im['id']] = []
            continue
        b, s, _ = weighted_boxes_fusion(caixas, scores, labels, weights=pesos,
                                        iou_thr=iou_wbf, skip_box_thr=skip)
        b = b * escala
        ordem = np.argsort(-s)[:100]            # COCO avalia ate 100 deteccoes por imagem
        resultado[im['id']] = [list(b[k]) + [float(s[k])] for k in ordem]
    return resultado


def avaliar_coco(coco_gt, det):
    """mAP, mAP@0,50 e mAP@0,75 do COCO."""
    res = [{'image_id': iid, 'category_id': 1, 'score': d[4],
            'bbox': [d[0], d[1], d[2] - d[0], d[3] - d[1]]}
           for iid, lista in det.items() for d in lista]
    if not res:
        return 0.0, 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):  # o pycocotools imprime muito
        coco_dt = coco_gt.loadRes(res)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[1], ev.stats[2]


def main():
    p = argparse.ArgumentParser(description='Ensemble Faster R-CNN + FCOS + ATSS com WBF (VinDr-Mammo)')
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--pesos', type=float, nargs=3, default=PESOS_PADRAO,
                   help='peso de cada modelo no WBF (Faster, FCOS, ATSS)')
    p.add_argument('--iou-wbf', type=float, default=IOU_WBF_PADRAO,
                   help='IoU minima para fundir caixas (padrao do WBF: 0,55)')
    p.add_argument('--skip', type=float, default=SKIP_PADRAO,
                   help='ignora deteccoes abaixo deste score antes da fusao')
    p.add_argument('--thr', type=float, default=THR_PADRAO, help='score minimo para TPR e FPPI')
    p.add_argument('--refazer', action='store_true')
    args = p.parse_args()

    cfg_base = Config.fromfile(MODELOS[0][1])
    ann = os.path.join(cfg_base.ANNOTS_DIR, f'{args.split}.json')
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(ann)
    imagens = coco_gt.dataset['images']
    gt = {im['id']: [] for im in imagens}
    for a in coco_gt.dataset['annotations']:
        x, y, w, h = a['bbox']
        gt[a['image_id']].append([x, y, x + w, y + h])

    deteccoes = rodar_modelos(cfg_base, args.split, args.refazer)
    deteccoes['ENSEMBLE (WBF)'] = fundir(deteccoes, imagens, args.pesos, args.iou_wbf, args.skip)

    print(f'\nConjunto: {args.split} | {len(imagens)} imagens | '
          f'{sum(len(v) for v in gt.values())} massas')
    print(f'WBF: pesos {args.pesos} | IoU {args.iou_wbf} | skip {args.skip} | '
          f'TPR/FPPI com score >= {args.thr} e IoU >= 0,5\n')
    print(f'{"Modelo":16s} {"mAP@0,50":>9s} {"mAP":>7s} {"mAP@0,75":>9s} {"TPR":>7s} {"FPPI":>6s}')
    linhas = []
    for nome, det in deteccoes.items():
        m, m50, m75 = avaliar_coco(coco_gt, det)
        tpr, fppi = tpr_fppi(gt, det, args.thr)
        print(f'{nome:16s} {m50 * 100:8.1f}% {m * 100:6.1f}% {m75 * 100:8.1f}% {tpr * 100:6.1f}% {fppi:6.2f}')
        linhas.append(dict(modelo=nome, mAP_50=round(m50, 4), mAP=round(m, 4), mAP_75=round(m75, 4),
                           TPR=round(tpr, 4), FPPI=round(fppi, 3)))

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    saida = os.path.join(PASTA_SAIDA, f'resultado_{args.split}.json')
    with open(saida, 'w', encoding='utf-8') as f:
        json.dump(dict(split=args.split, pesos=args.pesos, iou_wbf=args.iou_wbf, skip=args.skip,
                       thr=args.thr, resultados=linhas), f, indent=2, ensure_ascii=False)
    print('\nResultado salvo em:', saida)


if __name__ == '__main__':
    main()
