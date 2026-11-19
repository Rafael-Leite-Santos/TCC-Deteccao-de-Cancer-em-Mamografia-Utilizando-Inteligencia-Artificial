"""
Ponto 9 do orientador (e-mail de 15/11/2026): avaliar os nossos modelos com o MESMO
protocolo de avaliacao do Digital Eye (Terzi et al., 2025), lido no codigo original
(`mass_inference.py` e `utils/all_utils.py` do repositorio cbddobvyz/digitaleye-mammography,
copia em C:/Users/rl-sa/Digital_Eye_DICOM-PNG, mmdet 2.28.2), em 17/11/2026:

  1. NMS por classe com IoU 0,1 (`apply_nms` -> ensemble_boxes.nms, --nms_iou_threshold 0.1)
  2. descarta deteccoes com score <= 0,2 (`apply_nms`: `if s[n] > scr_thr`, --confidence_threshold 0.2)
  3. AP estilo VOC, area sob a curva precisao x revocacao (mmdet 2.x `eval_map`, mode='area'),
     com IoU de acerto 0,1 (`model_evals` -> dataset.evaluate(metric='mAP', iou_thr=--ap_threshold 0.1))

Aqui o passo 3 e reimplementado (as duas versoes do mmdet sao incompativeis): cada deteccao,
em ordem de score, casa com a massa de maior IoU ainda livre (IoU >= limiar) = VP; senao FP;
AP = area sob a curva com precisao interpolada (envelope), como o `average_precision` do mmdet 2.x.
Coordenadas continuas (use_legacy_coordinate=False, o padrao do mmdet 2.28).

Usa o cache de deteccoes do analise_erros.py (score >= 0,05, sem filtros); nao roda o modelo.
Para comparar, calcula tambem as variantes (IoU de acerto 0,5; sem o NMS/limiar do Digital Eye).
O codigo do Digital Eye NAO calcula TPR/FPPI (nao ha essas metricas em mass_inference.py/all_utils.py);
aqui o TPR/FPPI e o do metricas.py, so como referencia.

Uso (da raiz do clone):
  python avaliar_protocolo_digitaleye.py                                   # Faster de referencia
  python avaliar_protocolo_digitaleye.py --work_dir work_dirs/sementes/r50rot_s0
"""
import argparse
import json
import os
import sys

import numpy as np

from metricas import iou, tpr_fppi

ANNOTS_DIR = 'annotations_VinDR'
WORK_DIR_PADRAO = 'work_dirs/VinDR_faster_rcnn_40ep_parada_map50'


def ler_gt(split):
    with open(os.path.join(ANNOTS_DIR, split + '.json'), encoding='utf-8') as f:
        d = json.load(f)
    gt = {im['file_name']: [] for im in d['images']}
    nome = {im['id']: im['file_name'] for im in d['images']}
    for a in d['annotations']:
        x, y, w, h = a['bbox']
        gt[nome[a['image_id']]].append([x, y, x + w, y + h])
    return gt


def nms(dets, iou_thr):
    """NMS classico (uma classe): mantem a de maior score e remove as com IoU > iou_thr."""
    restantes = sorted(dets, key=lambda d: -d[4])
    mantidas = []
    while restantes:
        d = restantes.pop(0)
        mantidas.append(d)
        restantes = [r for r in restantes if iou(d, r) <= iou_thr]
    return mantidas


def ap_voc_area(gt, det, iou_thr):
    """AP (area sob a curva P x R, precisao interpolada), todas as imagens juntas."""
    n_gt = sum(len(v) for v in gt.values())
    todas = [(d[4], fn, d) for fn, ds in det.items() for d in ds]
    todas.sort(key=lambda t: -t[0])
    usadas = {fn: set() for fn in gt}
    tp = np.zeros(len(todas))
    for i, (_, fn, d) in enumerate(todas):
        melhor, k = 0.0, -1
        for j, g in enumerate(gt.get(fn, [])):
            v = iou(d, g)
            if v > melhor:
                melhor, k = v, j
        # como no tpfp_default do mmdet: casa com a massa de maior IoU; se ja usada, e FP
        if melhor >= iou_thr and k not in usadas[fn]:
            usadas[fn].add(k)
            tp[i] = 1
    fp = 1 - tp
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, np.finfo(float).eps)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', default=WORK_DIR_PADRAO, help='pasta do modelo com analise_erros/deteccoes_<split>.json')
    ap.add_argument('--split', default='val,test')
    args = ap.parse_args()

    nome = os.path.basename(os.path.normpath(args.work_dir))
    saida = os.path.join('analises', 'protocolo_digitaleye')
    os.makedirs(saida, exist_ok=True)
    out = [f'# Protocolo de avaliacao do Digital Eye aplicado a `{nome}`', '',
           'Protocolo (lido no codigo do Digital Eye em 17/11/2026): NMS IoU 0,1 -> score > 0,2 -> AP VOC (area) com IoU 0,1.',
           'Gerado por avaliar_protocolo_digitaleye.py a partir do cache do analise_erros.py. Numeros agregados.',
           'Os numeros do Digital Eye (KETEM, privado) continuam NAO comparaveis: o dataset e outro.', '']
    for split in [s.strip() for s in args.split.split(',') if s.strip()]:
        cache = os.path.join(args.work_dir, 'analise_erros', f'deteccoes_{split}.json')
        if not os.path.isfile(cache):
            raise SystemExit(f'Cache nao encontrado: {cache}\nRode antes: python analise_erros.py --config <config do modelo>')
        with open(cache, encoding='utf-8') as f:
            c = json.load(f)
        gt = ler_gt(split)
        bruto = {fn: c['det'].get(fn, []) for fn in gt}
        de = {fn: [d for d in nms(ds, 0.1) if d[4] > 0.2] for fn, ds in bruto.items()}
        out.append(f'## {split} ({len(gt)} imagens, {sum(len(v) for v in gt.values())} massas)\n')
        out.append('| Variante | AP (IoU 0,1) | AP (IoU 0,5) | TPR / FPPI (IoU 0,1, score > 0,2) | TPR / FPPI (IoU 0,5, score > 0,2) |')
        out.append('|---|---|---|---|---|')
        for rot, det in (('Protocolo Digital Eye (NMS 0,1 + score > 0,2)', de),
                         ('Sem NMS 0,1 / limiar (saida do nosso modelo, score >= 0,05)', bruto)):
            a1, a5 = ap_voc_area(gt, det, 0.1), ap_voc_area(gt, det, 0.5)
            t1, f1 = tpr_fppi(gt, det, 0.2 + 1e-9, iou_min=0.1)
            t5, f5 = tpr_fppi(gt, det, 0.2 + 1e-9, iou_min=0.5)
            out.append(f'| {rot} | {100 * a1:.1f}% | {100 * a5:.1f}% | {100 * t1:.1f}% / {f1:.2f} | {100 * t5:.1f}% / {f5:.2f} |')
        out.append('')
    caminho = os.path.join(saida, f'protocolo_{nome}.md')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print(f'Relatorio: {caminho}')


if __name__ == '__main__':
    sys.exit(main())
