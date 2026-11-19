"""
reavaliar_iou.py - reavalia os modelos do ensemble com varios limiares de IoU, sem retreinar

Le as deteccoes ja salvas pelo VinDR_ensemble.py (resultados_ensemble/deteccoes_<split>.json),
refaz a fusao WBF com os mesmos parametros padrao e calcula, para cada IoU da lista:
  - AP@IoU (COCO, area "all", ate 100 deteccoes por imagem);
  - TPR e FPPI com score >= --thr (mesmo tpr_fppi() do metricas.py, so muda o iou_min).

Objetivo: aproximar o criterio do Digital Eye (IoU 0,1) e ver o quanto os numeros
dependem do IoU. Mesmo com IoU 0,1, os resultados NAO sao comparaveis com os do
Digital Eye (dataset KETEM privado, outras classes, outro NMS).

Nao altera nenhum arquivo existente: so le o cache e grava
resultados_ensemble/reavaliacao_iou_<split>.json.

Uso:
  python reavaliar_iou.py                    teste oficial, IoU 0,1 0,2 0,3 0,4 0,5
  python reavaliar_iou.py --split val
  python reavaliar_iou.py --ious 0.1 0.5
"""
import argparse
import contextlib
import io
import json
import os

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from VinDR_ensemble import (PASTA_SAIDA, PESOS_PADRAO, IOU_WBF_PADRAO, SKIP_PADRAO,
                            fundir)
from metricas import THR_PADRAO, tpr_fppi

ANNOTS_DIR = 'annotations_VinDR'   # mesmo ANNOTS_DIR de configs/VinDR_faster_rcnn.py


def ap_por_iou(coco_gt, det, ious):
    """AP (COCO, area all, maxDets 100) para cada IoU da lista: {iou: ap}."""
    res = [{'image_id': iid, 'category_id': 1, 'score': d[4],
            'bbox': [d[0], d[1], d[2] - d[0], d[3] - d[1]]}
           for iid, lista in det.items() for d in lista]
    if not res:
        return {t: 0.0 for t in ious}
    with contextlib.redirect_stdout(io.StringIO()):  # o pycocotools imprime muito
        coco_dt = coco_gt.loadRes(res)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.iouThrs = np.array(ious, dtype=np.float64)
        ev.evaluate()
        ev.accumulate()
    # precision: [T limiares de IoU, R pontos de recall, K classes, A areas, M maxDets]
    # area 0 = "all"; maxDets -1 = 100
    prec = ev.eval['precision'][:, :, 0, 0, -1]
    saida = {}
    for i, t in enumerate(ious):
        p = prec[i][prec[i] > -1]
        saida[t] = float(p.mean()) if p.size else 0.0
    return saida


def main():
    p = argparse.ArgumentParser(description='Varredura de IoU sobre o cache do ensemble (sem retreinar)')
    p.add_argument('--split', default='test', choices=['test', 'val'])
    p.add_argument('--ious', type=float, nargs='+', default=[0.1, 0.2, 0.3, 0.4, 0.5])
    p.add_argument('--thr', type=float, default=THR_PADRAO, help='score minimo para TPR e FPPI')
    args = p.parse_args()

    cache = os.path.join(PASTA_SAIDA, f'deteccoes_{args.split}.json')
    if not os.path.isfile(cache):
        raise SystemExit(f'Cache nao encontrado: {cache}. Rode antes: python VinDR_ensemble.py --split {args.split}')
    with open(cache, encoding='utf-8') as f:
        deteccoes = {n: {int(k): v for k, v in d.items()} for n, d in json.load(f).items()
                     if n != '_checkpoints'}  # registro dos checkpoints (VinDR_ensemble, 18/11)

    ann = os.path.join(ANNOTS_DIR, f'{args.split}.json')
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(ann)
    imagens = coco_gt.dataset['images']
    gt = {im['id']: [] for im in imagens}
    for a in coco_gt.dataset['annotations']:
        x, y, w, h = a['bbox']
        gt[a['image_id']].append([x, y, x + w, y + h])

    # mesma fusao do VinDR_ensemble.py (parametros padrao)
    deteccoes['ENSEMBLE (WBF)'] = fundir(deteccoes, imagens, PESOS_PADRAO, IOU_WBF_PADRAO, SKIP_PADRAO)

    print(f'\nConjunto: {args.split} | {len(imagens)} imagens | {sum(len(v) for v in gt.values())} massas')
    print(f'Cache: {cache}')
    print(f'WBF: pesos {PESOS_PADRAO} | IoU {IOU_WBF_PADRAO} | skip {SKIP_PADRAO} | '
          f'TPR/FPPI com score >= {args.thr}\n')
    print(f'{"Modelo":16s} {"IoU":>5s} {"AP@IoU":>8s} {"TPR":>7s} {"FPPI":>6s}')

    linhas = []
    for nome, det in deteccoes.items():
        aps = ap_por_iou(coco_gt, det, args.ious)
        for t in args.ious:
            tpr, fppi = tpr_fppi(gt, det, args.thr, iou_min=t)
            print(f'{nome:16s} {t:5.2f} {aps[t] * 100:7.1f}% {tpr * 100:6.1f}% {fppi:6.2f}')
            linhas.append(dict(modelo=nome, iou=t, AP=round(aps[t], 4),
                               TPR=round(tpr, 4), FPPI=round(fppi, 3)))
        print()

    saida = os.path.join(PASTA_SAIDA, f'reavaliacao_iou_{args.split}.json')
    with open(saida, 'w', encoding='utf-8') as f:
        json.dump(dict(split=args.split, cache=cache, pesos=PESOS_PADRAO, iou_wbf=IOU_WBF_PADRAO,
                       skip=SKIP_PADRAO, thr=args.thr, ious=args.ious, resultados=linhas),
                  f, indent=2, ensure_ascii=False)
    print('Resultado salvo em:', saida)


if __name__ == '__main__':
    main()
