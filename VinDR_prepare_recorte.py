"""
VinDR_prepare_recorte.py - anotacoes do VinDr-Mammo com a imagem recortada na mama

Le as anotacoes ja geradas pelo VinDR_prepare.py (train/val/test.json) e cria
uma copia em PASTA_SAIDA em que:
  * cada imagem ganha o campo 'crop' = [x0, y0, x1, y1], o retangulo da mama
    (mesma mascara usada no treino: windowing + maior regiao conectada);
  * width/height passam a ser os do recorte;
  * as caixas das massas sao deslocadas para as coordenadas do recorte.

A divisao treino/validacao/teste NAO muda: sao as mesmas imagens, para que a
comparacao com o treino sem recorte meca so o efeito do recorte.

No fim mostra quantos pixels cada imagem ocupa depois do Resize, com e sem
recorte, e quanto a mama ganha em resolucao.

Uso:  python VinDR_prepare_recorte.py      (leva alguns minutos: le ~1.100 DICOMs)
"""
import json
import os
import time

import numpy as np

from VinDR_pipeline import carregar_vindr, retangulo_da_mama

# ---- AJUSTE AQUI ----
VINDR_ROOT     = r'data/VinDr-Mammo'
PASTA_ENTRADA  = r'annotations_VinDR'          # saida do VinDR_prepare.py
PASTA_SAIDA    = r'annotations_VinDR_recorte'
MARGEM_PX      = 20                 # folga em volta da mama, em pixels da imagem original
ESCALA_ATUAL   = (1024, 1024)       # Resize do treino sem recorte
ESCALA_RECORTE = (1600, 1024)       # Resize proposto para o treino com recorte
# ---------------------


def fator_resize(h, w, escala):
    """Fator do Resize(keep_ratio=True) do MMDetection: lado maior <= escala[0], menor <= escala[1]."""
    lado_maior, lado_menor = max(escala), min(escala)
    return min(lado_maior / max(h, w), lado_menor / min(h, w))


def recortar_split(nome, pasta_imgs, stats):
    with open(os.path.join(PASTA_ENTRADA, nome + '.json'), encoding='utf-8') as f:
        coco = json.load(f)
    anns_por_img = {}
    for a in coco['annotations']:
        anns_por_img.setdefault(a['image_id'], []).append(a)

    novas_anns = []
    t0 = time.time()
    for i, img in enumerate(coco['images'], 1):
        caminho = os.path.join(pasta_imgs, img['file_name'])
        _, mascara = carregar_vindr(caminho)
        x0, y0, x1, y1 = retangulo_da_mama(mascara, MARGEM_PX)
        H, W = mascara.shape
        if (W, H) != (img['width'], img['height']):
            print(f"  AVISO {img['file_name']}: tamanho do DICOM {W}x{H} difere do JSON "
                  f"{img['width']}x{img['height']}")

        for a in anns_por_img.get(img['id'], []):
            bx, by, bw, bh = a['bbox']
            nx1, ny1 = max(bx, x0), max(by, y0)
            nx2, ny2 = min(bx + bw, x1), min(by + bh, y1)
            area_antes = bw * bh
            area_depois = max(0, nx2 - nx1) * max(0, ny2 - ny1)
            stats['caixas'] += 1
            if area_depois < 0.999 * area_antes:
                stats['caixas_cortadas'] += 1
                stats['menor_fracao'] = min(stats['menor_fracao'], area_depois / area_antes)
            if area_depois < 0.5 * area_antes:          # massa quase toda fora do recorte
                stats['caixas_perdidas'] += 1
                print(f"  AVISO {img['file_name']}: caixa {a['bbox']} ficou com "
                      f"{area_depois / area_antes:.0%} da area dentro do recorte (descartada)")
                continue
            w2, h2 = nx2 - nx1, ny2 - ny1
            novas_anns.append(dict(a, bbox=[round(nx1 - x0, 1), round(ny1 - y0, 1),
                                            round(w2, 1), round(h2, 1)],
                                   area=round(w2 * h2, 1)))

        ch, cw = y1 - y0, x1 - x0
        f_atual = fator_resize(H, W, ESCALA_ATUAL)
        f_rec = fator_resize(ch, cw, ESCALA_RECORTE)
        stats['fracao_area'].append(ch * cw / (H * W))
        stats['px_atual'].append(H * W * f_atual ** 2)
        stats['px_recorte'].append(ch * cw * f_rec ** 2)
        stats['ganho'].append(f_rec / f_atual)

        img['crop'] = [x0, y0, x1, y1]
        img['width'], img['height'] = cw, ch
        if i % 100 == 0:
            print(f'  {nome}: {i}/{len(coco["images"])} imagens ({time.time() - t0:.0f} s)')

    coco['annotations'] = novas_anns
    with open(os.path.join(PASTA_SAIDA, nome + '.json'), 'w', encoding='utf-8') as f:
        json.dump(coco, f)
    print(f'{nome:5s}: {len(coco["images"]):5d} imagens | {len(novas_anns):5d} massas')


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    pasta_imgs = os.path.join(VINDR_ROOT, 'images')
    stats = dict(caixas=0, caixas_cortadas=0, caixas_perdidas=0, menor_fracao=1.0,
                 fracao_area=[], px_atual=[], px_recorte=[], ganho=[])
    for nome in ('train', 'val', 'test'):
        recortar_split(nome, pasta_imgs, stats)

    m = lambda k: float(np.mean(stats[k]))
    print('\n---- Resumo ----')
    print(f"Area do recorte: {m('fracao_area'):.0%} da imagem em media "
          f"(min {min(stats['fracao_area']):.0%}, max {max(stats['fracao_area']):.0%})")
    print(f"Pixels por imagem depois do Resize: sem recorte {ESCALA_ATUAL} = {m('px_atual') / 1e6:.2f} M | "
          f"com recorte {ESCALA_RECORTE} = {m('px_recorte') / 1e6:.2f} M")
    print(f"Resolucao da mama (fator linear): {m('ganho'):.2f}x em media "
          f"(min {min(stats['ganho']):.2f}x, max {max(stats['ganho']):.2f}x)")
    print(f"Caixas: {stats['caixas']} | encostadas na borda do recorte: {stats['caixas_cortadas']}"
          + (f" (menor fracao mantida {stats['menor_fracao']:.0%})" if stats['caixas_cortadas'] else '')
          + f" | descartadas: {stats['caixas_perdidas']}")
    print('Anotacoes salvas em:', PASTA_SAIDA)


if __name__ == '__main__':
    main()
