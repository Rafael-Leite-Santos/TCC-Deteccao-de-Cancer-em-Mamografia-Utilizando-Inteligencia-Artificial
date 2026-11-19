"""
VinDR_inference.py - deteccao de massas em DICOM com o modelo do VinDr

Uso:
  python VinDR_inference.py caminho\\imagem.dicom
  python VinDR_inference.py caminho\\pasta          (todas as .dicom/.dcm, inclusive subpastas)

Opcoes:
  --checkpoint arquivo.pth   padrao: o best_*.pth mais recente do work_dir do config
  --config arquivo.py        padrao: configs/VinDR_faster_rcnn.py
  --thr 0.3                  score minimo para mostrar uma deteccao
  --saida pasta              padrao: resultados_VinDR
  --filtros                  descarta deteccoes com centro fora da mama ou caixa
                             maior que 30% da imagem (padrao: sem filtros, como
                             na tabela do README / VinDR_ensemble.py)

Para cada imagem salva um PNG com as deteccoes (vermelho) e, se a imagem
estiver no finding_annotations.csv, as massas anotadas pelos radiologistas
(verde). Ao fim, grava deteccoes.csv e, quando houver anotacao, mostra a
sensibilidade (TPR) e os falsos positivos por imagem (FPPI) com IoU >= 0,5,
calculados por metricas.tpr_fppi (pareamento 1 para 1, o mesmo do ensemble).

O pre-processamento e a mesma funcao do treino (carregar_vindr): windowing
do DICOM, correcao de MONOCHROME1 e isolamento da mama. Se o config usa
LoadVindrDicom(recortar=True), a imagem e recortada na mama como no treino e
as caixas voltam para as coordenadas da imagem original.
"""
import argparse
import ast
import csv
import glob
import os

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector

from VinDR_pipeline import carregar_vindr, retangulo_da_mama  # tambem registra LoadVindrDicom
from metricas import THR_PADRAO, tpr_fppi

CONFIG_PADRAO = 'configs/VinDR_faster_rcnn.py'
VERMELHO, VERDE = (0, 0, 255), (0, 200, 0)


def listar_dicoms(caminho):
    if os.path.isfile(caminho):
        return [caminho]
    arquivos = []
    for ext in ('*.dicom', '*.dcm'):
        arquivos += glob.glob(os.path.join(caminho, '**', ext), recursive=True)
    return sorted(arquivos)


def melhor_checkpoint(work_dir):
    candidatos = glob.glob(os.path.join(work_dir, 'best_*.pth'))
    if not candidatos:
        raise SystemExit(f'Nenhum best_*.pth em {work_dir}. Use --checkpoint.')
    return max(candidatos, key=os.path.getmtime)


def ler_anotacoes(vindr_root):
    """image_id -> lista de caixas de massa [x1, y1, x2, y2] (None se nao houver CSV)."""
    caminho = os.path.join(vindr_root or '', 'finding_annotations.csv')
    if not os.path.isfile(caminho):
        return None
    gt = {}
    with open(caminho, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            caixas = gt.setdefault(row['image_id'].strip(), [])
            try:
                cats = ast.literal_eval(row['finding_categories'])
            except (ValueError, SyntaxError):
                cats = [row['finding_categories']]
            if 'Mass' in cats and row['xmin'].strip():
                caixas.append([float(row[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')])
    return gt


def usa_recorte(model):
    """True se o config do modelo usa LoadVindrDicom(recortar=True)."""
    try:
        return bool(model.cfg.test_dataloader.dataset.pipeline[0].get('recortar', False))
    except (AttributeError, KeyError, IndexError):
        return False


def detectar(model, caminho, thr, recortar=False, filtros=True):
    """Roda o modelo e, com filtros=True, descarta as deteccoes com centro fora da
    mama ou caixa maior que 30% da imagem. Devolve (imagem, deteccoes, descartadas).
    As caixas saem sempre nas coordenadas da imagem original."""
    img, mascara = carregar_vindr(caminho)
    h, w = img.shape[:2]

    # Com recorte, o modelo recebe so o retangulo da mama; (x0, y0) leva as
    # caixas de volta para a imagem original
    x0 = y0 = 0
    entrada = img
    if recortar:
        x0, y0, x1, y1 = retangulo_da_mama(mascara)
        entrada = img[y0:y1, x0:x1].copy()

    # Passa a imagem ja pre-processada; o MMDetection troca o LoadVindrDicom
    # do pipeline por LoadImageFromNDArray e aplica o mesmo Resize do treino
    pred = inference_detector(model, entrada).pred_instances
    bboxes = pred.bboxes.cpu().numpy() + np.array([x0, y0, x0, y0], dtype=np.float32)
    scores = pred.scores.cpu().numpy()
    deteccoes, descartadas = filtrar(bboxes, scores, mascara, thr, filtros)
    return img, deteccoes, descartadas


def filtrar(bboxes, scores, mascara, thr, filtros=True):
    """Fica com as caixas de score >= thr e, com filtros=True, descarta as de centro
    fora da mama ou maiores que 30% da imagem. Usado pelo detectar() e pelo
    detectar_ensemble() do VinDR_ensemble.py. Devolve (deteccoes, descartadas)."""
    h, w = mascara.shape[:2]
    deteccoes, descartadas = [], 0
    for (x1, y1, x2, y2), s in zip(bboxes, scores):
        if s < thr:
            continue
        if not filtros:
            deteccoes.append(([float(x1), float(y1), float(x2), float(y2)], float(s)))
            continue
        cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
        cy = int(np.clip((y1 + y2) / 2, 0, h - 1))
        if mascara[cy, cx] == 0:                       # centro fora da mama
            descartadas += 1
            continue
        if (x2 - x1) * (y2 - y1) > 0.30 * h * w:       # caixa grande demais
            descartadas += 1
            continue
        deteccoes.append(([float(x1), float(y1), float(x2), float(y2)], float(s)))
    return deteccoes, descartadas


def desenhar(img, deteccoes, caixas_gt):
    out = img.copy()
    esp = max(3, int(max(out.shape[:2]) / 500))        # espessura proporcional
    fonte = max(0.8, max(out.shape[:2]) / 2000)
    for x1, y1, x2, y2 in caixas_gt or []:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), VERDE, esp)
    for (x1, y1, x2, y2), s in deteccoes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), VERMELHO, esp)
        cv2.putText(out, f'mass {s:.2f}', (int(x1), max(int(y1) - 3 * esp, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, fonte, VERMELHO, max(2, esp // 2))
    return out


def main():
    p = argparse.ArgumentParser(description='Inferencia de massas em DICOM (VinDr)')
    p.add_argument('entrada', help='arquivo .dicom/.dcm ou pasta')
    p.add_argument('--config', default=CONFIG_PADRAO)
    p.add_argument('--checkpoint')
    p.add_argument('--thr', type=float, default=THR_PADRAO)
    p.add_argument('--saida', default='resultados_VinDR')
    p.add_argument('--filtros', action='store_true',
                   help='descarta deteccoes fora da mama ou maiores que 30%% da imagem')
    args = p.parse_args()

    arquivos = listar_dicoms(args.entrada)
    if not arquivos:
        raise SystemExit(f'Nenhum .dicom/.dcm encontrado em {args.entrada}')

    cfg = Config.fromfile(args.config)
    checkpoint = args.checkpoint or melhor_checkpoint(cfg.work_dir)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Modelo: {checkpoint} ({device})')
    model = init_detector(args.config, checkpoint, device=device)
    recortar = usa_recorte(model)
    print('Recorte da mama:', 'sim' if recortar else 'nao')
    print('Filtros (fora da mama / caixa grande):', 'sim' if args.filtros else 'nao')

    gt = ler_anotacoes(cfg.get('VINDR_ROOT'))
    os.makedirs(args.saida, exist_ok=True)

    linhas = []
    gt_metricas, det_metricas = {}, {}   # so as imagens com anotacao

    for i, caminho in enumerate(arquivos, 1):
        image_id = os.path.splitext(os.path.basename(caminho))[0]
        try:
            img, deteccoes, descartadas = detectar(model, caminho, args.thr, recortar, args.filtros)
        except Exception as e:
            print(f'[{i}/{len(arquivos)}] {image_id}: ERRO - {e}')
            continue

        caixas_gt = gt.get(image_id) if gt is not None else None
        print(f'[{i}/{len(arquivos)}] {image_id}: {len(deteccoes)} deteccoes'
              + (f' ({descartadas} descartadas fora da mama/grandes)' if descartadas else '')
              + (f' | {len(caixas_gt)} massas anotadas' if caixas_gt is not None else ''))

        if caixas_gt is not None:
            gt_metricas[image_id] = caixas_gt
            det_metricas[image_id] = [b + [s] for b, s in deteccoes]

        for (x1, y1, x2, y2), s in deteccoes:
            linhas.append([image_id, f'{s:.4f}', int(x1), int(y1), int(x2), int(y2)])

        cv2.imwrite(os.path.join(args.saida, image_id + '_deteccao.png'),
                    desenhar(img, deteccoes, caixas_gt))

    with open(os.path.join(args.saida, 'deteccoes.csv'), 'w', newline='', encoding='utf-8') as f:
        escritor = csv.writer(f)
        escritor.writerow(['image_id', 'score', 'x1', 'y1', 'x2', 'y2'])
        escritor.writerows(linhas)

    print(f'\nResultados em: {os.path.abspath(args.saida)}')
    if gt_metricas:
        n_gt = sum(len(c) for c in gt_metricas.values())
        tpr, fppi = tpr_fppi(gt_metricas, det_metricas, args.thr)
        print(f'Com anotacao: {len(gt_metricas)} imagens | {n_gt} massas')
        print(f'  Sensibilidade (TPR, IoU>=0,5, score>={args.thr}, pareamento 1 para 1): '
              f'{tpr * 100:.1f}% ({round(tpr * n_gt)}/{n_gt})')
        print(f'  Falsos positivos por imagem (FPPI): {fppi:.2f}')


if __name__ == '__main__':
    main()
