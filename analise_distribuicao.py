"""
Ponto 1 do orientador (e-mail de 15/11/2026): os conjuntos de treino, validacao e
teste tem distribuicoes parecidas?

Compara, para as imagens COM MASSA usadas no treino (annotations_VinDR/*.json):
  - massas por imagem; tamanho das caixas (px e mm); faixas de tamanho do COCO
  - BI-RADS da massa, BI-RADS da mama, densidade mamaria, lado e incidencia (CSV)
  - fabricante / modelo do equipamento e tamanho do pixel (cabecalho DICOM)
e testa as diferencas (Kolmogorov-Smirnov para numeros, qui-quadrado para categorias).

So grava numeros agregados (nenhum ID de imagem/estudo, nenhum dado de paciente),
em analises/distribuicao/.

Uso (da raiz do clone, com o venv ativo):
  python analise_distribuicao.py              # le tambem o cabecalho dos DICOMs (~1-3 min)
  python analise_distribuicao.py --sem-dicom  # so JSON + CSV (segundos)
"""
import argparse
import ast
import csv
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

VINDR_ROOT = 'data/VinDr-Mammo'
IMAGES_ROOT = VINDR_ROOT + '/images'
ANNOTS_DIR = 'annotations_VinDR'
SAIDA = os.path.join('analises', 'distribuicao')
SPLITS = ('train', 'val', 'test')


def ler_json(split):
    with open(os.path.join(ANNOTS_DIR, split + '.json'), encoding='utf-8') as f:
        d = json.load(f)
    imgs = {im['id']: im for im in d['images']}
    caixas = defaultdict(list)
    for a in d['annotations']:
        caixas[a['image_id']].append(a['bbox'])          # [x, y, w, h] em px do DICOM
    return imgs, caixas


def categorias(txt):
    try:
        return [c.strip() for c in ast.literal_eval(txt)]
    except Exception:
        return [txt.strip()]


def ler_csv_massas():
    """image_id -> dados da imagem; e lista de linhas de massa (para BI-RADS da massa)."""
    caminho = os.path.join(VINDR_ROOT, 'finding_annotations.csv')
    if not os.path.isfile(caminho):
        print(f'AVISO: {caminho} nao encontrado (HD D: conectado?). Pulando dados do CSV.')
        return {}, defaultdict(list), []
    por_imagem, massas = {}, defaultdict(list)
    with open(caminho, newline='', encoding='utf-8') as f:
        leitor = csv.DictReader(f)
        colunas = leitor.fieldnames
        for row in leitor:
            iid = row['image_id'].strip()
            por_imagem.setdefault(iid, {k: (row.get(k) or '').strip() for k in
                                        ('laterality', 'view_position', 'breast_birads',
                                         'breast_density')})
            if 'Mass' in categorias(row['finding_categories']) and row['xmin'].strip():
                massas[iid].append((row.get('finding_birads') or '').strip())
    return por_imagem, massas, colunas


def ler_dicom(caminho):
    import pydicom
    try:
        ds = pydicom.dcmread(caminho, stop_before_pixels=True,
                             specific_tags=['Manufacturer', 'ManufacturerModelName',
                                            'PixelSpacing', 'ImagerPixelSpacing'])
    except Exception as e:
        return {'erro': type(e).__name__}
    esp = ds.get('PixelSpacing') or ds.get('ImagerPixelSpacing')
    try:
        esp = float(esp[0]) if esp is not None else None
    except Exception:
        esp = None
    return {'fabricante': str(ds.get('Manufacturer', '') or 'desconhecido').strip(),
            'modelo': str(ds.get('ManufacturerModelName', '') or 'desconhecido').strip(),
            'pixel_mm': esp}


def faixa_coco(area):
    # mesmas faixas do CocoMetric (area em px da imagem original, pois o val_pipeline
    # carrega as anotacoes depois do Resize): pequena < 32^2 <= media < 96^2 <= grande
    if area < 32 ** 2:
        return 'pequena (<32px)'
    if area < 96 ** 2:
        return 'media (32-96px)'
    return 'grande (>=96px)'


def resumo_num(v):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if v.size == 0:
        return 'sem dados'
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return (f'n={v.size}  media={v.mean():.1f}  mediana={med:.1f}  '
            f'Q1-Q3={q1:.1f}-{q3:.1f}  min={v.min():.1f}  max={v.max():.1f}')


def tabela_cat(nome, dados, out):
    """dados: split -> lista de rotulos. Imprime % por split e qui-quadrado."""
    rotulos = sorted({r for s in SPLITS for r in dados[s]}, key=str)
    if not rotulos:
        return
    out.append(f'\n## {nome}\n')
    out.append('| categoria | ' + ' | '.join(f'{s} (n={len(dados[s])})' for s in SPLITS) + ' |')
    out.append('|---' * (len(SPLITS) + 1) + '|')
    cont = {s: Counter(dados[s]) for s in SPLITS}
    for r in rotulos:
        cel = []
        for s in SPLITS:
            n = len(dados[s]) or 1
            cel.append(f'{cont[s][r]} ({100 * cont[s][r] / n:.1f}%)')
        out.append(f'| {r or "(vazio)"} | ' + ' | '.join(cel) + ' |')
    try:
        from scipy.stats import chi2_contingency
        m = np.array([[cont[s][r] for r in rotulos] for s in SPLITS])
        m = m[:, m.sum(axis=0) > 0]
        if m.shape[1] > 1:
            chi2, p, gl, esp = chi2_contingency(m)
            aviso = ' (atencao: ha caselas com esperado < 5; p aproximado)' if (esp < 5).any() else ''
            out.append(f'\nQui-quadrado (3 conjuntos): chi2={chi2:.2f}, gl={gl}, p={p:.3f}{aviso}')
    except Exception as e:
        out.append(f'\n(qui-quadrado nao calculado: {e})')


def tabela_num(nome, dados, out):
    out.append(f'\n## {nome}\n')
    for s in SPLITS:
        out.append(f'- {s}: {resumo_num(dados[s])}')
    try:
        from scipy.stats import ks_2samp
        for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
            x = [v for v in dados[a] if v is not None]
            y = [v for v in dados[b] if v is not None]
            if x and y:
                r = ks_2samp(x, y)
                out.append(f'- KS {a} x {b}: D={r.statistic:.3f}, p={r.pvalue:.3f}')
    except Exception as e:
        out.append(f'(KS nao calculado: {e})')


def grafico(dados_lado_px, dados_lado_mm):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'AVISO: grafico nao gerado ({e})')
        return
    for nome, dados, rot in (('lado_px', dados_lado_px, 'Lado da caixa (raiz da area, px do DICOM)'),
                             ('lado_mm', dados_lado_mm, 'Lado da caixa (raiz da area, mm)')):
        series = {s: [v for v in dados[s] if v is not None] for s in SPLITS}
        if not any(series.values()):
            continue
        todos = np.concatenate([np.asarray(v) for v in series.values() if v])
        bins = np.linspace(0, np.percentile(todos, 99), 30)
        fig, ax = plt.subplots(figsize=(7, 4))
        for s in SPLITS:
            if series[s]:
                ax.hist(series[s], bins=bins, density=True, histtype='step', linewidth=2,
                        label=f'{s} (n={len(series[s])})')
        if nome == 'lado_px':
            for lim in (32, 96):
                ax.axvline(lim, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel(rot)
        ax.set_ylabel('Densidade (normalizada por conjunto)')
        ax.set_title('Distribuicao do tamanho das massas por conjunto (VinDr-Mammo)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(SAIDA, f'tamanho_massas_{nome}.png'), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sem-dicom', action='store_true', help='nao le o cabecalho dos DICOMs')
    args = ap.parse_args()
    os.makedirs(SAIDA, exist_ok=True)

    csv_img, csv_massas, colunas = ler_csv_massas()
    if colunas:
        print('Colunas do finding_annotations.csv:', ', '.join(colunas))

    d = {k: {s: [] for s in SPLITS} for k in (
        'massas_por_imagem', 'lado_px', 'lado_mm', 'faixa_coco', 'birads_massa',
        'birads_mama', 'densidade', 'lado', 'incidencia', 'fabricante', 'modelo',
        'pixel_mm', 'altura_img', 'largura_img')}
    falhas_dicom = Counter()

    for s in SPLITS:
        imgs, caixas = ler_json(s)
        for img_id, im in imgs.items():
            iid = os.path.splitext(os.path.basename(im['file_name']))[0]
            bbs = caixas.get(img_id, [])
            d['massas_por_imagem'][s].append(len(bbs))
            d['altura_img'][s].append(im['height'])
            d['largura_img'][s].append(im['width'])
            meta = csv_img.get(iid, {})
            d['birads_mama'][s].append(meta.get('breast_birads', ''))
            d['densidade'][s].append(meta.get('breast_density', ''))
            d['lado'][s].append(meta.get('laterality', ''))
            d['incidencia'][s].append(meta.get('view_position', ''))
            d['birads_massa'][s].extend(csv_massas.get(iid, []))

            px = None
            if not args.sem_dicom:
                info = ler_dicom(os.path.join(IMAGES_ROOT, im['file_name']))
                if 'erro' in info:
                    falhas_dicom[info['erro']] += 1
                else:
                    d['fabricante'][s].append(info['fabricante'])
                    d['modelo'][s].append(info['modelo'])
                    d['pixel_mm'][s].append(info['pixel_mm'])
                    px = info['pixel_mm']
            for x, y, w, h in bbs:
                lado = float(np.sqrt(w * h))
                d['lado_px'][s].append(lado)
                d['lado_mm'][s].append(lado * px if px else None)
                d['faixa_coco'][s].append(faixa_coco(w * h))
        print(f'{s}: {len(imgs)} imagens, {sum(len(v) for v in caixas.values())} massas')

    # conferencia: o CSV deve ter as mesmas massas que os JSONs
    for s in SPLITS:
        n_json = len(d['lado_px'][s])
        n_csv = len(d['birads_massa'][s])
        if csv_img and n_json != n_csv:
            print(f'AVISO {s}: {n_json} massas no JSON x {n_csv} no CSV (BI-RADS da massa pode estar desalinhado)')

    out = ['# Distribuicao dos conjuntos (VinDr-Mammo, so imagens com massa)',
           '',
           'Gerado por analise_distribuicao.py. Numeros agregados; nenhum ID ou dado de paciente.',
           'KS = Kolmogorov-Smirnov (2 amostras); p < 0,05 sugere distribuicoes diferentes.',
           'Com n pequeno (val: 92 imagens / 97 massas) os testes tem pouco poder: p alto nao prova igualdade.']
    tabela_num('Massas por imagem', d['massas_por_imagem'], out)
    tabela_num('Tamanho da massa: raiz da area da caixa (px do DICOM)', d['lado_px'], out)
    if any(v for s in SPLITS for v in d['lado_mm'][s] if v is not None):
        tabela_num('Tamanho da massa: raiz da area da caixa (mm)', d['lado_mm'], out)
    tabela_cat('Faixas de tamanho do COCO (as usadas em mAP_s / mAP_m / mAP_l)', d['faixa_coco'], out)
    tabela_cat('BI-RADS da massa (finding_birads)', d['birads_massa'], out)
    tabela_cat('BI-RADS da mama (breast_birads)', d['birads_mama'], out)
    tabela_cat('Densidade mamaria (breast_density)', d['densidade'], out)
    tabela_cat('Lado', d['lado'], out)
    tabela_cat('Incidencia', d['incidencia'], out)
    if not args.sem_dicom:
        tabela_cat('Fabricante (DICOM Manufacturer)', d['fabricante'], out)
        tabela_cat('Modelo do equipamento (DICOM ManufacturerModelName)', d['modelo'], out)
        tabela_num('Tamanho do pixel (mm)', d['pixel_mm'], out)
        if falhas_dicom:
            out.append(f'\nFalhas ao ler cabecalho DICOM: {dict(falhas_dicom)}')
    tabela_num('Altura da imagem (px)', d['altura_img'], out)
    tabela_num('Largura da imagem (px)', d['largura_img'], out)

    caminho = os.path.join(SAIDA, 'relatorio_distribuicao.md')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    grafico(d['lado_px'], d['lado_mm'])
    print('\n'.join(out))
    print(f'\nRelatorio: {caminho}')
    print(f'Graficos:  {SAIDA}{os.sep}tamanho_massas_lado_px.png / _lado_mm.png')


if __name__ == '__main__':
    sys.exit(main())
