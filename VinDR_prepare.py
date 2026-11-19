"""
VinDR_prepare.py - gera as anotacoes COCO (train/val/test) do VinDr-Mammo

Le o finding_annotations.csv e cria, na pasta SAIDA:
  train.json  - imagens do split "training" (menos as de validacao)
  val.json    - FRACAO_VAL dos estudos do "training", separados por estudo
  test.json   - imagens do split "test" oficial (usado so na avaliacao final)

Como no CBIS-DDSM, entram as imagens que tem pelo menos uma massa
('Mass' em finding_categories). A separacao treino/validacao e feita por
estudo (study_id): as 4 incidencias de um exame ficam sempre do mesmo lado,
e o modelo nunca e validado na outra incidencia da mesma mama.

Diferente do CBIS, as caixas ja vem prontas no CSV (xmin, ymin, xmax, ymax),
em pixels da imagem original. Nao ha mascara para recortar.

Uso:  python VinDR_prepare.py
"""
import ast
import csv
import json
import os
import random
from collections import defaultdict

# ---- AJUSTE AQUI ----
VINDR_ROOT = r'data/VinDr-Mammo'   # pasta com finding_annotations.csv e images/
SAIDA      = r'annotations_VinDR'
FRACAO_VAL = 0.10   # 10% dos estudos do treino vao para validacao
SEMENTE    = 42     # fixa a divisao treino/validacao (reprodutibilidade)

# Imagens sem nenhum achado ("No Finding") adicionadas so ao TREINO, como
# exemplos negativos, na proporcao de N negativas por imagem com massa.
# 0 = igual a metodologia do CBIS (so imagens com massa).
NEGATIVOS_POR_POSITIVO = 0.0
# ---------------------


def categorias(texto):
    """"['Mass', 'Skin Retraction']" -> ['Mass', 'Skin Retraction']"""
    try:
        return [c.strip() for c in ast.literal_eval(texto)]
    except (ValueError, SyntaxError):
        return [texto.strip()]


def ler_csv(caminho_csv):
    imagens = {}                   # image_id -> dados da imagem
    caixas = defaultdict(list)     # image_id -> caixas de massa
    tem_achado = set()             # imagens com qualquer achado (nao so massa)

    with open(caminho_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            image_id = row['image_id'].strip()
            imagens.setdefault(image_id, {
                'study_id': row['study_id'].strip(),
                'width': int(float(row['width'])),
                'height': int(float(row['height'])),
                'split': row['split'].strip().lower(),
            })

            cats = categorias(row['finding_categories'])
            if cats != ['No Finding']:
                tem_achado.add(image_id)

            if 'Mass' not in cats or not row['xmin'].strip():
                continue

            img = imagens[image_id]
            x1 = max(0.0, float(row['xmin']))
            y1 = max(0.0, float(row['ymin']))
            x2 = min(float(img['width']), float(row['xmax']))
            y2 = min(float(img['height']), float(row['ymax']))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            caixas[image_id].append([x1, y1, x2 - x1, y2 - y1])

    negativas = [i for i in imagens if i not in tem_achado]
    return imagens, caixas, negativas


def montar_coco(ids, imagens, caixas):
    coco = {'images': [], 'annotations': [],
            'categories': [{'id': 1, 'name': 'mass'}]}
    ann_id = 1
    for img_id, image_id in enumerate(sorted(ids), start=1):
        img = imagens[image_id]
        coco['images'].append({
            'id': img_id,
            'file_name': f"{img['study_id']}/{image_id}.dicom",
            'width': img['width'],
            'height': img['height'],
        })
        for x, y, w, h in caixas.get(image_id, []):
            coco['annotations'].append({
                'id': ann_id, 'image_id': img_id, 'category_id': 1,
                'bbox': [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                'area': round(w * h, 1), 'iscrowd': 0,
            })
            ann_id += 1
    return coco


def main():
    caminho_csv = os.path.join(VINDR_ROOT, 'finding_annotations.csv')
    pasta_imgs = os.path.join(VINDR_ROOT, 'images')
    imagens, caixas, negativas = ler_csv(caminho_csv)

    # So mantem imagens cujo arquivo existe (util se o download estiver incompleto)
    def existe(image_id):
        img = imagens[image_id]
        return os.path.isfile(os.path.join(pasta_imgs, img['study_id'], image_id + '.dicom'))

    positivas = [i for i in caixas if caixas[i]]
    faltando = {i for i in positivas if not existe(i)}
    positivas = [i for i in positivas if i not in faltando]

    # Divisao: test oficial / treino e validacao por estudo
    teste = [i for i in positivas if imagens[i]['split'] == 'test']
    treino_total = [i for i in positivas if imagens[i]['split'] != 'test']

    estudos = sorted({imagens[i]['study_id'] for i in treino_total})
    random.Random(SEMENTE).shuffle(estudos)
    estudos_val = set(estudos[:max(1, round(len(estudos) * FRACAO_VAL))])
    val = [i for i in treino_total if imagens[i]['study_id'] in estudos_val]
    treino = [i for i in treino_total if imagens[i]['study_id'] not in estudos_val]

    # Negativas opcionais (so treino, de estudos fora da validacao)
    n_neg = 0
    if NEGATIVOS_POR_POSITIVO > 0:
        pool = [i for i in negativas
                if imagens[i]['split'] != 'test'
                and imagens[i]['study_id'] not in estudos_val and existe(i)]
        random.Random(SEMENTE).shuffle(pool)
        extra = pool[:round(len(treino) * NEGATIVOS_POR_POSITIVO)]
        treino += extra
        n_neg = len(extra)

    os.makedirs(SAIDA, exist_ok=True)
    print(f'Imagens no CSV: {len(imagens)} | com massa: {len(positivas) + len(faltando)}')
    if faltando:
        print(f'AVISO: {len(faltando)} imagens com massa nao encontradas em {pasta_imgs} (ignoradas)')

    for nome, ids in (('train', treino), ('val', val), ('test', teste)):
        coco = montar_coco(ids, imagens, caixas)
        with open(os.path.join(SAIDA, nome + '.json'), 'w', encoding='utf-8') as f:
            json.dump(coco, f)
        n_est = len({imagens[i]['study_id'] for i in ids})
        print(f'{nome:5s}: {len(coco["images"]):5d} imagens | '
              f'{len(coco["annotations"]):5d} massas | {n_est:5d} estudos')
    if n_neg:
        print(f'       (treino inclui {n_neg} imagens negativas)')
    print('Anotacoes salvas em:', SAIDA)


if __name__ == '__main__':
    main()
