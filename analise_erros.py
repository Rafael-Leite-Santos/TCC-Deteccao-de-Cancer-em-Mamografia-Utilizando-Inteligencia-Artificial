"""
Ponto 8 do orientador (e-mail de 15/11/2026) e item 18 do contexto: analise dos
tipos de erro do detector, massa por massa e caixa por caixa.

Para cada massa anotada (score >= --thr, pareamento 1 para 1 igual ao metricas.py):
  acerto            pareada com IoU >= 0,5
  erro_localizacao  sem par, mas alguma caixa com 0,1 <= IoU < 0,5 (achou o lugar, caixa ruim)
  perdida_limiar    nenhuma caixa >= thr, mas existe caixa com IoU >= 0,5 e score < thr
  perdida           nada perto (IoU < 0,1 com tudo que passou do limiar)
Para cada caixa prevista sem par (falso positivo):
  fp_duplicata      IoU >= 0,5 com uma massa que ja foi pareada (2a caixa na mesma massa)
  fp_localizacao    0,1 <= IoU < 0,5 com alguma massa
  fp_fundo          IoU < 0,1 com todas as massas
Tudo separado por faixa de tamanho da massa. Tambem: TPR/FPPI com IoU 0,5 e 0,1,
razao de area (prevista/anotada) e deslocamento do centro nos acertos e erros de
localizacao, e grafico IoU x score por massa.

As deteccoes (score >= 0,05, sem filtros, como na avaliacao do treino) ficam em cache
dentro do work_dir do modelo (pasta ignorada pelo git); rodar de novo so refaz a
analise, sem GPU. O relatorio em analises/erros/ tem apenas numeros agregados.

Uso (da raiz do clone, com o venv ativo):
  python analise_erros.py                                  # Faster de referencia, val e test
  python analise_erros.py --config configs/VinDR_faster_rcnn_rot.py
  python analise_erros.py --thr 0.5 --split val
  python analise_erros.py --refazer                        # ignora o cache e roda o modelo
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from metricas import THR_PADRAO, iou, tpr_fppi

ANNOTS_DIR = 'annotations_VinDR'
IMAGES_ROOT = 'data/VinDr-Mammo/images'
CONFIG_PADRAO = 'configs/VinDR_faster_rcnn.py'
THR_CACHE = 0.05            # igual ao score_thr do test_cfg: guarda tudo que a avaliacao COCO ve
FAIXAS = ((0, 150), (150, 250), (250, 350), (350, 1e9))   # lado da caixa (raiz da area), px do DICOM


def nome_faixa(lado):
    for a, b in FAIXAS:
        if a <= lado < b:
            return f'{int(a)}-{int(b)} px' if b < 1e9 else f'>={int(a)} px'
    return '?'


def ler_gt(split):
    with open(os.path.join(ANNOTS_DIR, split + '.json'), encoding='utf-8') as f:
        d = json.load(f)
    gt = {im['file_name']: [] for im in d['images']}
    nome = {im['id']: im['file_name'] for im in d['images']}
    for a in d['annotations']:
        x, y, w, h = a['bbox']
        gt[nome[a['image_id']]].append([x, y, x + w, y + h])
    return gt


def rodar_modelo(config, checkpoint, gt, cache):
    import torch
    from mmdet.apis import init_detector
    from VinDR_inference import detectar, usa_recorte
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'Modelo: {checkpoint} ({device})')
    model = init_detector(config, checkpoint, device=device)
    recortar = usa_recorte(model)
    det, t0 = {}, time.time()
    for i, fn in enumerate(gt, 1):
        _, dets, _ = detectar(model, os.path.join(IMAGES_ROOT, fn), THR_CACHE,
                              recortar=recortar, filtros=False)
        det[fn] = [b + [s] for b, s in dets]
        if i % 25 == 0 or i == len(gt):
            print(f'  {i}/{len(gt)} imagens ({time.time() - t0:.0f} s)')
    with open(cache, 'w', encoding='utf-8') as f:
        json.dump({'checkpoint': checkpoint, 'thr_cache': THR_CACHE, 'det': det}, f)
    return det


def analisar(gt, det, thr):
    massas, fps = [], []          # uma entrada por massa anotada / por falso positivo
    for fn, caixas in gt.items():
        todas = sorted(det.get(fn, []), key=lambda d: -d[4])
        dets = [d for d in todas if d[4] >= thr]
        usadas = {}
        livres_fp = []
        for d in dets:                          # mesmo pareamento do metricas.tpr_fppi
            melhor, k = 0.0, -1
            for j, g in enumerate(caixas):
                if j not in usadas:
                    v = iou(d, g)
                    if v > melhor:
                        melhor, k = v, j
            if melhor >= 0.5:
                usadas[k] = d
            else:
                livres_fp.append(d)
        for j, g in enumerate(caixas):
            lado = float(np.sqrt((g[2] - g[0]) * (g[3] - g[1])))
            ious = [iou(d, g) for d in dets]
            iou_max = max(ious) if ious else 0.0
            perto = [d for d in todas if iou(d, g) >= 0.5]
            score_max = max((d[4] for d in perto), default=0.0)   # melhor score com IoU >= 0,5
            if j in usadas:
                tipo, ref = 'acerto', usadas[j]
            elif iou_max >= 0.1:
                tipo, ref = 'erro_localizacao', dets[int(np.argmax(ious))]
            elif perto:
                tipo, ref = 'perdida_limiar', None
            else:
                tipo, ref = 'perdida', None
            info = {'tipo': tipo, 'faixa': nome_faixa(lado), 'lado': lado,
                    'iou_max': iou_max, 'score_iou50': score_max}
            if ref is not None:
                ag = (g[2] - g[0]) * (g[3] - g[1])
                ap = (ref[2] - ref[0]) * (ref[3] - ref[1])
                info['razao_area'] = ap / ag if ag > 0 else None
                cg = ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2)
                cp = ((ref[0] + ref[2]) / 2, (ref[1] + ref[3]) / 2)
                info['desloc_centro'] = float(np.hypot(cp[0] - cg[0], cp[1] - cg[1])) / max(lado, 1)
            massas.append(info)
        for d in livres_fp:
            ious = [iou(d, g) for g in caixas]
            m = max(ious) if ious else 0.0
            if m >= 0.5:
                tipo = 'fp_duplicata'
            elif m >= 0.1:
                tipo = 'fp_localizacao'
            else:
                tipo = 'fp_fundo'
            fps.append({'tipo': tipo, 'score': d[4]})
    return massas, fps


def pct(n, t):
    return f'{n} ({100 * n / t:.1f}%)' if t else '0'


def relatorio(split, gt, det, massas, fps, thr, out):
    n_img, n_m = len(gt), len(massas)
    out.append(f'\n# Conjunto: {split} ({n_img} imagens, {n_m} massas, limiar de score {thr})\n')
    for iou_min in (0.5, 0.1):
        tpr, fppi = tpr_fppi(gt, det, thr, iou_min=iou_min)
        out.append(f'- TPR/FPPI com IoU >= {iou_min}: TPR {100 * tpr:.1f}%, FPPI {fppi:.2f}')
    out.append('  (IoU 0,1 aqui e so o limiar de pareamento; NAO e ainda o protocolo do Digital Eye)')

    tipos_m = ('acerto', 'erro_localizacao', 'perdida_limiar', 'perdida')
    c = Counter(m['tipo'] for m in massas)
    out.append('\n## Massas anotadas por tipo\n')
    out.append('| tipo | n (%) |')
    out.append('|---|---|')
    for t in tipos_m:
        out.append(f'| {t} | {pct(c[t], n_m)} |')

    out.append('\n## Massas por faixa de tamanho (lado da caixa anotada)\n')
    out.append('| faixa | n | ' + ' | '.join(tipos_m) + ' |')
    out.append('|---' * (len(tipos_m) + 2) + '|')
    por_faixa = defaultdict(list)
    for m in massas:
        por_faixa[m['faixa']].append(m['tipo'])
    for a, b in FAIXAS:
        f = nome_faixa(a)
        lst = por_faixa.get(f, [])
        cf = Counter(lst)
        out.append(f'| {f} | {len(lst)} | ' + ' | '.join(pct(cf[t], len(lst)) for t in tipos_m) + ' |')

    cfp = Counter(f['tipo'] for f in fps)
    out.append(f'\n## Falsos positivos (score >= {thr}): {len(fps)} no total, {len(fps) / max(n_img, 1):.2f} por imagem\n')
    out.append('| tipo | n (%) |')
    out.append('|---|---|')
    for t in ('fp_duplicata', 'fp_localizacao', 'fp_fundo'):
        out.append(f'| {t} | {pct(cfp[t], len(fps))} |')

    out.append('\n## Tamanho e posicao das caixas previstas\n')
    for t in ('acerto', 'erro_localizacao'):
        ra = [m['razao_area'] for m in massas if m['tipo'] == t and m.get('razao_area')]
        dc = [m['desloc_centro'] for m in massas if m['tipo'] == t and m.get('desloc_centro') is not None]
        io = [m['iou_max'] for m in massas if m['tipo'] == t]
        if ra:
            out.append(f'- {t} (n={len(ra)}): IoU mediano {np.median(io):.2f}; '
                       f'razao de area prevista/anotada mediana {np.median(ra):.2f} '
                       f'(Q1-Q3 {np.percentile(ra, 25):.2f}-{np.percentile(ra, 75):.2f}; '
                       f'{sum(r > 1 for r in ra)} maiores, {sum(r < 1 for r in ra)} menores que a anotacao); '
                       f'deslocamento do centro mediano {np.median(dc):.2f} x lado da massa')
    perd = [m['score_iou50'] for m in massas if m['tipo'] == 'perdida_limiar']
    if perd:
        out.append(f'- perdidas pelo limiar: melhor score com IoU >= 0,5 mediano {np.median(perd):.2f} '
                   f'(min {min(perd):.2f}, max {max(perd):.2f})')


def grafico(resultados, saida, thr):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'AVISO: grafico nao gerado ({e})')
        return
    cores = {'acerto': 'tab:green', 'erro_localizacao': 'tab:orange',
             'perdida_limiar': 'tab:blue', 'perdida': 'tab:red'}
    fig, axs = plt.subplots(1, len(resultados), figsize=(6 * len(resultados), 4.5), squeeze=False)
    for ax, (split, massas) in zip(axs[0], resultados.items()):
        for t, cor in cores.items():
            pts = [(m['iou_max'], m['score_iou50']) for m in massas if m['tipo'] == t]
            if pts:
                x, y = zip(*pts)
                ax.scatter(x, y, s=18, alpha=0.75, color=cor, label=f'{t} ({len(pts)})')
        ax.axvline(0.5, color='gray', ls='--', lw=1)
        ax.axvline(0.1, color='gray', ls=':', lw=1)
        ax.axhline(thr, color='gray', ls='--', lw=1)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(f'Maior IoU entre a massa e uma caixa com score >= {thr}')
        ax.set_ylabel('Maior score entre as caixas com IoU >= 0,5')
        ax.set_title(f'{split}: uma marca por massa anotada')
        ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    caminho = os.path.join(saida, 'erros_por_massa.png')
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    print(f'Grafico:   {caminho}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=CONFIG_PADRAO)
    ap.add_argument('--checkpoint', default=None, help='padrao: best_*.pth mais recente do work_dir do config')
    ap.add_argument('--split', default='val,test')
    ap.add_argument('--thr', type=float, default=THR_PADRAO)
    ap.add_argument('--refazer', action='store_true', help='ignora o cache de deteccoes')
    args = ap.parse_args()

    from mmengine.config import Config
    work_dir = Config.fromfile(args.config).work_dir
    checkpoint = args.checkpoint
    if checkpoint is None:
        from VinDR_inference import melhor_checkpoint
        checkpoint = melhor_checkpoint(work_dir)
    nome_modelo = os.path.splitext(os.path.basename(args.config))[0]
    saida = os.path.join('analises', 'erros', nome_modelo)
    os.makedirs(saida, exist_ok=True)
    pasta_cache = os.path.join(work_dir, 'analise_erros')
    os.makedirs(pasta_cache, exist_ok=True)

    out = [f'# Analise de erros: {nome_modelo}', '',
           f'Checkpoint: `{checkpoint.replace(os.sep, "/")}`. Deteccoes sem filtros, como na avaliacao do treino.',
           'Gerado por analise_erros.py. Numeros agregados; nenhum ID ou dado de paciente.']
    resultados = {}
    for split in [s.strip() for s in args.split.split(',') if s.strip()]:
        gt = ler_gt(split)
        cache = os.path.join(pasta_cache, f'deteccoes_{split}.json')
        det = None
        if os.path.isfile(cache) and not args.refazer:
            with open(cache, encoding='utf-8') as f:
                c = json.load(f)
            if os.path.normcase(os.path.abspath(c['checkpoint'])) == os.path.normcase(os.path.abspath(checkpoint)):
                det = c['det']
                print(f'{split}: usando cache {cache}')
        if det is None:
            print(f'{split}: rodando o modelo em {len(gt)} imagens')
            det = rodar_modelo(args.config, checkpoint, gt, cache)
        massas, fps = analisar(gt, det, args.thr)
        relatorio(split, gt, det, massas, fps, args.thr, out)
        resultados[split] = massas

    caminho = os.path.join(saida, f'relatorio_erros_thr{args.thr:g}.md')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print(f'\nRelatorio: {caminho}')
    grafico(resultados, saida, args.thr)


if __name__ == '__main__':
    sys.exit(main())
