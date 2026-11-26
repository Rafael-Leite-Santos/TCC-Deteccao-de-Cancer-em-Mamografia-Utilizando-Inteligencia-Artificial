"""
avaliar_fp_normais.py - falsos positivos em exames NORMAIS do teste oficial

O teste usado em todos os treinos (annotations_VinDR/test.json) so tem imagens com
massa, entao o FPPI dele nao mostra o que o modelo faz num exame normal. Este script
roda cada modelo tambem em imagens do split "test" oficial sem nenhum achado
("No Finding") e mede:
  - FPPI em normais: caixas com score >= thr por imagem normal (toda caixa e FP);
  - % de imagens normais com pelo menos uma caixa (= 1 - especificidade por imagem);
  - TPR nas 237 massas do teste (igual ao metricas.py), no mesmo limiar;
  - TPR com FPPI em normais de 0,25 / 0,5 / 1,0 (varrendo o limiar, estilo FROC).
Sem treinar: so inferencia. Deteccoes sem filtros (score >= 0,05), como na avaliacao
do treino, em cache no work_dir de cada modelo (pasta ignorada pelo git).

As normais sao sorteadas uma vez (semente fixa) e sao as MESMAS para todos os modelos.
O relatorio em analises/fp_normais/ tem so numeros agregados (nenhum ID).

Uso (da raiz do clone, com o venv ativo):
  python avaliar_fp_normais.py                  # r50aug_s0..2 x r50augneg_s0..2, 1000 normais
  python avaliar_fp_normais.py --max 0          # todas as normais do teste
  python avaliar_fp_normais.py --configs configs/sementes/r50aug_s2.py
Checkpoints: work_dir do config; se nao houver best_*.pth la, procura em
D:\\TCC_backup\\work_dirs (ou em --backup).
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np

import VinDR_prepare as prep
from analise_erros import ler_gt, rodar_modelo, THR_CACHE
from metricas import tpr_fppi

CONFIGS_PADRAO = ([f'configs/sementes/r50aug_s{i}.py' for i in range(3)] +
                  [f'configs/sementes/r50augneg_s{i}.py' for i in range(3)])
BACKUP_PADRAO = r'D:\TCC_backup\work_dirs'
LIMIARES = (0.3, 0.5, 0.7)
ALVOS_FPPI = (0.25, 0.5, 1.0)


def normais_do_teste(maximo, semente):
    caminho_csv = os.path.join(prep.VINDR_ROOT, 'finding_annotations.csv')
    imagens, _, negativas = prep.ler_csv(caminho_csv)
    pasta = os.path.join(prep.VINDR_ROOT, 'images')
    nomes = []
    for i in sorted(negativas):
        if imagens[i]['split'] != 'test':
            continue
        fn = f"{imagens[i]['study_id']}/{i}.dicom"
        if os.path.isfile(os.path.join(pasta, fn)):
            nomes.append(fn)
    total = len(nomes)
    random.Random(semente).shuffle(nomes)
    if maximo:
        nomes = nomes[:maximo]
    return sorted(nomes), total


def achar_checkpoint(config, backup):
    from VinDR_ensemble import pasta_do_modelo
    for pasta in (pasta_do_modelo(config), pasta_do_modelo(config, backup)):
        cands = glob.glob(os.path.join(pasta, 'best_*.pth'))
        if cands:
            return max(cands, key=os.path.getmtime), pasta
    raise SystemExit(f'Nenhum best_*.pth para {config} (nem em {backup}).')


def medir(gt_pos, normais, det, thr):
    tpr, fppi_pos = tpr_fppi(gt_pos, det, thr)
    n_caixas = [sum(1 for d in det.get(fn, []) if d[4] >= thr) for fn in normais]
    return {'tpr': tpr, 'fppi_pos': fppi_pos,
            'fppi_normais': float(np.mean(n_caixas)),
            'img_normais_com_fp': float(np.mean([n > 0 for n in n_caixas]))}


def tpr_em_fppi(gt_pos, normais, det, alvo):
    """Maior TPR cujo FPPI em normais fica <= alvo (varrendo o limiar de 0,05 a 0,99)."""
    melhor = 0.0
    for thr in np.arange(THR_CACHE, 1.0, 0.01):
        m = medir(gt_pos, normais, det, float(thr))
        if m['fppi_normais'] <= alvo:
            melhor = max(melhor, m['tpr'])
    return melhor


def curva(gt_pos, normais, det):
    pts = []
    for thr in np.arange(THR_CACHE, 1.0, 0.02):
        m = medir(gt_pos, normais, det, float(thr))
        pts.append((m['fppi_normais'], m['tpr']))
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', default=CONFIGS_PADRAO)
    ap.add_argument('--max', type=int, default=1000, help='normais sorteadas (0 = todas)')
    ap.add_argument('--semente', type=int, default=42)
    ap.add_argument('--backup', default=BACKUP_PADRAO)
    ap.add_argument('--refazer', action='store_true')
    args = ap.parse_args()

    gt_pos = ler_gt('test')                       # 219 imagens com massa
    normais, total_normais = normais_do_teste(args.max, args.semente)
    print(f'Teste: {len(gt_pos)} imagens com massa | normais: {len(normais)} de {total_normais}')
    if set(normais) & set(gt_pos):
        raise SystemExit('ERRO: imagem normal tambem esta no test.json.')
    todas = dict(gt_pos)
    todas.update({fn: [] for fn in normais})

    resultados = {}
    for config in args.configs:
        nome = os.path.splitext(os.path.basename(config))[0]
        ckpt, pasta = achar_checkpoint(config, args.backup)
        cache_dir = os.path.join(pasta, 'analise_fp_normais')
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f'deteccoes_test_e_normais{args.max}_s{args.semente}.json')
        det = None
        if os.path.isfile(cache) and not args.refazer:
            with open(cache, encoding='utf-8') as f:
                c = json.load(f)
            if (os.path.normcase(os.path.abspath(c['checkpoint'])) == os.path.normcase(os.path.abspath(ckpt))
                    and set(c['det']) == set(todas)):
                det = c['det']
                print(f'{nome}: usando cache')
        if det is None:
            print(f'{nome}: {len(todas)} imagens, checkpoint {ckpt}')
            det = rodar_modelo(config, ckpt, todas, cache)
        r = {'checkpoint': os.path.basename(ckpt)}
        for thr in LIMIARES:
            r[thr] = medir(gt_pos, normais, det, thr)
        for alvo in ALVOS_FPPI:
            r[f'tpr@{alvo}'] = tpr_em_fppi(gt_pos, normais, det, alvo)
        r['curva'] = curva(gt_pos, normais, det)
        resultados[nome] = r
        m = r[0.3]
        print(f"  thr 0,3: TPR {m['tpr']:.1%} | FPPI massas {m['fppi_pos']:.2f} | "
              f"FPPI normais {m['fppi_normais']:.2f} | normais com FP {m['img_normais_com_fp']:.1%}")

    relatorio(resultados, len(normais), total_normais, args)


def grupo(nome):
    return nome.rsplit('_s', 1)[0]


def relatorio(res, n_normais, total_normais, args):
    saida = os.path.join('analises', 'fp_normais')
    os.makedirs(saida, exist_ok=True)
    pct = lambda v: f'{100 * v:.1f}%'.replace('.', ',')
    num = lambda v: f'{v:.2f}'.replace('.', ',')
    out = ['# Falsos positivos em exames normais (teste oficial do VinDr-Mammo)', '',
           f'Imagens: 219 com massa (237 massas) + {n_normais} normais ("No Finding") sorteadas '
           f'(semente {args.semente}) de {total_normais} normais do split test. Mesmas imagens para todos os modelos.',
           'Deteccoes sem filtros. TPR com IoU >= 0,5 e pareamento 1 para 1 (metricas.py). '
           'Gerado por avaliar_fp_normais.py; numeros agregados, nenhum ID.', '']
    for thr in LIMIARES:
        out += [f'## Limiar de score {num(thr)}', '',
                '| Modelo | Checkpoint | TPR (massas) | FPPI nas imagens com massa | FPPI em normais | Normais com >= 1 caixa |',
                '|---|---|---|---|---|---|']
        for nome, r in res.items():
            m = r[thr]
            out.append(f"| {nome} | {r['checkpoint']} | {pct(m['tpr'])} | {num(m['fppi_pos'])} | "
                       f"{num(m['fppi_normais'])} | {pct(m['img_normais_com_fp'])} |")
        out.append('')
    out += ['## TPR com FPPI em normais fixo (varrendo o limiar)', '',
            '| Modelo | ' + ' | '.join(f'TPR @ FPPI {num(a)}' for a in ALVOS_FPPI) + ' |',
            '|---|' + '---|' * len(ALVOS_FPPI)]
    for nome, r in res.items():
        out.append(f'| {nome} | ' + ' | '.join(pct(r[f"tpr@{a}"]) for a in ALVOS_FPPI) + ' |')
    out.append('')

    grupos = {}
    for nome, r in res.items():
        grupos.setdefault(grupo(nome), []).append(r)
    if len(grupos) >= 2:
        out += ['## Media ± desvio padrao por configuracao (limiar 0,3) e teste t de Welch', '']
        metricas = [('TPR', lambda r: r[0.3]['tpr'], pct), ('FPPI em normais', lambda r: r[0.3]['fppi_normais'], num),
                    ('Normais com caixa', lambda r: r[0.3]['img_normais_com_fp'], pct),
                    ('TPR @ FPPI 0,5', lambda r: r['tpr@0.5'], pct), ('TPR @ FPPI 1,0', lambda r: r['tpr@1.0'], pct)]
        nomes_g = list(grupos)
        out += ['| Metrica | ' + ' | '.join(f'{g} (n={len(grupos[g])})' for g in nomes_g) + ' | p (Welch, 1a x 2a) |',
                '|---|' + '---|' * (len(nomes_g) + 1)]
        try:
            from scipy import stats
        except Exception:
            stats = None
        for rot, f, fmt in metricas:
            vals = [np.array([f(r) for r in grupos[g]]) for g in nomes_g]
            cel = [f'{fmt(v.mean())} ± {fmt(v.std(ddof=1)) if len(v) > 1 else "-"}' for v in vals]
            p = '-'
            if stats and len(vals[0]) > 1 and len(vals[1]) > 1:
                p = f'{stats.ttest_ind(vals[0], vals[1], equal_var=False).pvalue:.3f}'.replace('.', ',')
            out.append(f'| {rot} | ' + ' | '.join(cel) + f' | {p} |')
        out.append('')

    caminho = os.path.join(saida, 'resumo_fp_normais.md')
    with open(caminho, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print('\n'.join(out))
    print(f'Relatorio: {caminho}')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        cores = {}
        for nome, r in res.items():
            g = grupo(nome)
            cor = cores.setdefault(g, f'C{len(cores)}')
            x, y = zip(*r['curva'])
            ax.plot(x, [100 * v for v in y], color=cor, alpha=0.8,
                    label=g if nome.endswith('_s0') else None)
        ax.set_xlim(0, 3)
        ax.set_xlabel('Falsos positivos por imagem normal')
        ax.set_ylabel('TPR nas massas do teste (%)')
        ax.set_title('TPR x FP por imagem normal (uma curva por execucao)')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(saida, 'froc_normais.png'), dpi=150)
        plt.close(fig)
        print(f"Grafico:   {os.path.join(saida, 'froc_normais.png')}")
    except Exception as e:
        print(f'AVISO: grafico nao gerado ({e})')


if __name__ == '__main__':
    sys.exit(main())
