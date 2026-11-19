"""
Ponto 7 do orientador (e-mail de 15/11/2026): o desempenho muda com o equipamento
(fabricante/modelo), a densidade mamaria, o BI-RADS ou o tamanho da massa?

Usa o cache de deteccoes do analise_erros.py (nao roda o modelo), o finding_annotations.csv
e o cabecalho dos DICOMs (fabricante/modelo; nao le pixels nem dados de paciente).
Por subgrupo: imagens, massas, TPR e FPPI (score >= --thr, IoU >= 0,5, pareamento do
metricas.py) e AP VOC com IoU 0,5. Teste qui-quadrado para "massa achada x perdida" entre
subgrupos (e Fisher quando houver so 2 grupos). Grupos pequenos: numeros so indicativos.

Uso (da raiz do clone, com o venv ativo e o HD D: conectado):
  python analise_subgrupos.py                                         # Faster de referencia
  python analise_subgrupos.py --work_dir work_dirs/sementes/r50rot_s0 --split test
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

from metricas import THR_PADRAO, iou, tpr_fppi
from analise_distribuicao import ler_csv_massas, ler_dicom, categorias, VINDR_ROOT, IMAGES_ROOT
from avaliar_protocolo_digitaleye import ler_gt, ap_voc_area, WORK_DIR_PADRAO

import csv


def birads_por_massa():
    """image_id -> lista de (caixa x1,y1,x2,y2, finding_birads) das massas do CSV."""
    caminho = os.path.join(VINDR_ROOT, 'finding_annotations.csv')
    res = defaultdict(list)
    with open(caminho, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if 'Mass' in categorias(row['finding_categories']) and row['xmin'].strip():
                cx = [float(row[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
                res[row['image_id'].strip()].append((cx, (row.get('finding_birads') or '').strip()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work_dir', default=WORK_DIR_PADRAO)
    ap.add_argument('--split', default='val,test')
    ap.add_argument('--thr', type=float, default=THR_PADRAO)
    args = ap.parse_args()
    try:
        from scipy.stats import chi2_contingency, fisher_exact
    except Exception:
        chi2_contingency = fisher_exact = None

    csv_img, _, _ = ler_csv_massas()
    if not csv_img:
        raise SystemExit('Sem o finding_annotations.csv (HD D: conectado?).')
    massas_csv = birads_por_massa()
    nome = os.path.basename(os.path.normpath(args.work_dir))
    saida = os.path.join('analises', 'subgrupos')
    os.makedirs(saida, exist_ok=True)
    out = [f'# Desempenho por subgrupo: `{nome}`', '',
           f'TPR e FPPI com score >= {args.thr} e IoU >= 0,5 (metricas.py); AP VOC com IoU 0,5 (todas as deteccoes do cache, score >= 0,05).',
           'Gerado por analise_subgrupos.py. Numeros agregados; nenhum ID ou dado de paciente.',
           'Grupos com poucas massas: numeros apenas indicativos. p = qui-quadrado (achada x perdida entre grupos).', '']

    for split in [s.strip() for s in args.split.split(',') if s.strip()]:
        cache = os.path.join(args.work_dir, 'analise_erros', f'deteccoes_{split}.json')
        if not os.path.isfile(cache):
            raise SystemExit(f'Cache nao encontrado: {cache}\nRode antes: python analise_erros.py --config <config do modelo>')
        with open(cache, encoding='utf-8') as f:
            det = json.load(f)['det']
        gt = ler_gt(split)

        # atributos por imagem
        atrib = {}
        print(f'{split}: lendo cabecalhos DICOM de {len(gt)} imagens...')
        for fn in gt:
            iid = os.path.splitext(os.path.basename(fn))[0]
            meta = csv_img.get(iid, {})
            info = ler_dicom(os.path.join(IMAGES_ROOT, fn))
            atrib[fn] = {'Densidade mamaria': meta.get('breast_density', '') or '?',
                         'Fabricante': info.get('fabricante', '?'),
                         'Modelo do equipamento': info.get('modelo', '?'),
                         'Incidencia': meta.get('view_position', '') or '?'}

        # BI-RADS e tamanho por massa (casando a caixa do JSON com a do CSV)
        atrib_massa = {}
        for fn, caixas in gt.items():
            iid = os.path.splitext(os.path.basename(fn))[0]
            for j, g in enumerate(caixas):
                cand = massas_csv.get(iid, [])
                br = max(cand, key=lambda c: iou(c[0], g))[1] if cand else '?'
                lado = float(np.sqrt((g[2] - g[0]) * (g[3] - g[1])))
                faixa = '<150 px' if lado < 150 else '150-250 px' if lado < 250 else '250-350 px' if lado < 350 else '>=350 px'
                atrib_massa[(fn, j)] = {'BI-RADS da massa': br or '?', 'Tamanho da massa (lado)': faixa}

        out.append(f'\n# Conjunto: {split}\n')
        chaves_img = ['Densidade mamaria', 'Fabricante', 'Modelo do equipamento', 'Incidencia']
        chaves_massa = ['BI-RADS da massa', 'Tamanho da massa (lado)']
        for chave in chaves_img + chaves_massa:
            ORDEM = {'<150 px': 0, '150-250 px': 1, '250-350 px': 2, '>=350 px': 3}
            grupos = sorted({a[chave] for a in (atrib.values() if chave in chaves_img else atrib_massa.values())},
                            key=lambda g: (ORDEM.get(g, 9), g))
            out.append(f'## {chave}\n')
            out.append('| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |')
            out.append('|---|---|---|---|---|---|')
            tabela = []
            for gname in grupos:
                if chave in chaves_img:
                    fns = [fn for fn in gt if atrib[fn][chave] == gname]
                    g_sub = {fn: gt[fn] for fn in fns}
                else:   # por massa: mantem so as massas do grupo; FPPI nao se aplica
                    g_sub = {}
                    for fn, caixas in gt.items():
                        sel = [c for j, c in enumerate(caixas) if atrib_massa[(fn, j)][chave] == gname]
                        if sel:
                            g_sub[fn] = sel
                d_sub = {fn: det.get(fn, []) for fn in g_sub}
                n_m = sum(len(v) for v in g_sub.values())
                tpr, fppi = tpr_fppi(g_sub, d_sub, args.thr, iou_min=0.5)
                achadas = round(tpr * n_m)
                tabela.append([achadas, n_m - achadas])
                if chave in chaves_img:
                    apv = ap_voc_area(g_sub, d_sub, 0.5)
                    out.append(f'| {gname} | {len(g_sub)} | {n_m} | {100 * tpr:.1f}% | {fppi:.2f} | {100 * apv:.1f}% |')
                else:
                    out.append(f'| {gname} | {len(g_sub)} | {n_m} | {100 * tpr:.1f}% | (n/a) | (n/a) |')
            m = np.array(tabela)
            m = m[m.sum(axis=1) > 0]
            if chi2_contingency is not None and len(m) > 1 and (m[:, 1].sum() > 0):
                try:
                    if len(m) == 2:
                        p = fisher_exact(m)[1]
                        out.append(f'\nFisher (achada x perdida): p = {p:.3f}')
                    else:
                        chi2, p, gl, esp = chi2_contingency(m)
                        aviso = ' (caselas com esperado < 5: p aproximado)' if (esp < 5).any() else ''
                        out.append(f'\nQui-quadrado (achada x perdida): chi2 = {chi2:.2f}, gl = {gl}, p = {p:.3f}{aviso}')
                except Exception as e:
                    out.append(f'\n(teste nao calculado: {e})')
            out.append('')
    caminho = os.path.join(saida, f'subgrupos_{nome}.md')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print(f'Relatorio: {caminho}')


if __name__ == '__main__':
    sys.exit(main())
