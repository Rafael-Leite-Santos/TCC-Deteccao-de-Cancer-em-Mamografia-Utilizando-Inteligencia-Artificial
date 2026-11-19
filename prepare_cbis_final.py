import os
import json
import csv
import glob
import random
import pydicom
import numpy as np

# ---- AJUSTE AQUI (caminhos do CBIS-DDSM na sua maquina) ----
CSV_PATH    = r"data/CBIS-DDSM/mass_case_description_train_set.csv"
CBIS_ROOT   = r"data/CBIS-DDSM/cbis_ddsm"            # pastas das mascaras ROI
IMAGES_ROOT = r"data/CBIS-DDSM/imagens_completas"   # pastas Mass-Training_P_xxxxx_LADO_INCIDENCIA
OUTPUT_DIR  = r"annotations"
SPLIT_RATIO = 0.8

os.makedirs(OUTPUT_DIR, exist_ok=True)


def encontrar_dicom_full(images_root, patient_id, lado, view):
    for prefixo in ["Mass-Training", "Mass-Test"]:
        padrao = prefixo + "_P_" + patient_id + "_" + lado + "_" + view
        caminho = os.path.join(images_root, padrao)
        if os.path.exists(caminho):
            arquivos = glob.glob(os.path.join(caminho, "**", "*.dcm"), recursive=True)
            if arquivos:
                arquivos.sort(key=os.path.getsize, reverse=True)
                return arquivos[0]
    return None


def extrair_bbox_mascara_full(cbis_root, roi_path_csv):
    """
    O CBIS-DDSM tem dois arquivos na pasta ROI:
    - Arquivo MAIOR: mascara no tamanho da imagem full (correto!)
    - Arquivo menor: crop da regiao da massa
    Usamos o MAIOR para obter coordenadas corretas na imagem original.
    """
    nome_pasta = roi_path_csv.strip().split("/")[0]
    caminho_pasta = os.path.join(cbis_root, nome_pasta)

    if not os.path.exists(caminho_pasta):
        return None

    arquivos = glob.glob(os.path.join(caminho_pasta, "**", "*.dcm"), recursive=True)
    if not arquivos:
        return None

    # Pegar o MAIOR arquivo — e a mascara no tamanho da imagem full
    arquivos.sort(key=os.path.getsize, reverse=True)
    mascara_path = arquivos[0]

    try:
        ds = pydicom.dcmread(mascara_path)
        mascara = ds.pixel_array

        rows_nz = np.any(mascara > 0, axis=1)
        cols_nz = np.any(mascara > 0, axis=0)

        if not rows_nz.any():
            return None

        y1 = int(np.argmax(rows_nz))
        y2 = int(len(rows_nz) - np.argmax(rows_nz[::-1]) - 1)
        x1 = int(np.argmax(cols_nz))
        x2 = int(len(cols_nz) - np.argmax(cols_nz[::-1]) - 1)

        w = x2 - x1
        h = y2 - y1

        if w <= 0 or h <= 0:
            return None

        return [x1, y1, w, h]

    except Exception as e:
        return None


def ler_dimensoes(dicom_path):
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    return int(ds.Columns), int(ds.Rows)


def processar_csv(csv_path, images_root, cbis_root):
    imagens = {}
    anotacoes = []
    nao_encontrados = []
    sem_bbox = []
    img_id = 1
    ann_id = 1
    total = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            patient_id = row["patient_id"].replace("P_", "")
            lado = row["left or right breast"].strip().upper()
            view = row["image view"].strip().upper()
            roi_path = row["ROI mask file path"].strip()

            dicom_path = encontrar_dicom_full(images_root, patient_id, lado, view)
            if dicom_path is None:
                nao_encontrados.append("P_" + patient_id + "_" + lado + "_" + view)
                continue

            bbox = extrair_bbox_mascara_full(cbis_root, roi_path)
            if bbox is None:
                sem_bbox.append(roi_path[:50])
                continue

            rel_path = os.path.relpath(dicom_path, images_root)
            if rel_path not in imagens:
                width, height = ler_dimensoes(dicom_path)
                imagens[rel_path] = {
                    "id": img_id,
                    "file_name": rel_path,
                    "width": width,
                    "height": height
                }
                img_id += 1

            img_entry = imagens[rel_path]
            x, y, w, h = bbox
            anotacoes.append({
                "id": ann_id,
                "image_id": img_entry["id"],
                "category_id": 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1

            if ann_id % 100 == 0:
                print("Processados:", ann_id, "anotacoes...")

    print("Total no CSV:", total)
    print("Nao encontrados:", len(nao_encontrados))
    print("Sem bbox valida:", len(sem_bbox))
    return list(imagens.values()), anotacoes


def montar_coco(imagens, anotacoes):
    return {
        "info": {"description": "CBIS-DDSM — BBox ROI full-size correta"},
        "licenses": [],
        "categories": [{"id": 1, "name": "mass", "supercategory": "lesion"}],
        "images": imagens,
        "annotations": anotacoes
    }


def salvar_json(dados, caminho):
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2)
    print("Salvo:", caminho)


if __name__ == "__main__":
    print("Extraindo bboxes da mascara full-size...")
    imagens, anotacoes = processar_csv(CSV_PATH, IMAGES_ROOT, CBIS_ROOT)
    print("Imagens:", len(imagens))
    print("Anotacoes com bbox real:", len(anotacoes))

    # Verificar as primeiras 5 bboxes
    print("Primeiras 5 bboxes:")
    for ann in anotacoes[:5]:
        print(" ", ann["bbox"])

    random.seed(42)
    random.shuffle(imagens)
    corte = int(len(imagens) * SPLIT_RATIO)
    imgs_train = imagens[:corte]
    imgs_val   = imagens[corte:]
    ids_train = {img["id"] for img in imgs_train}
    ids_val   = {img["id"] for img in imgs_val}
    anns_train = [a for a in anotacoes if a["image_id"] in ids_train]
    anns_val   = [a for a in anotacoes if a["image_id"] in ids_val]

    salvar_json(montar_coco(imgs_train, anns_train), os.path.join(OUTPUT_DIR, "train.json"))
    salvar_json(montar_coco(imgs_val, anns_val), os.path.join(OUTPUT_DIR, "val.json"))
    print("Treino:", len(imgs_train), "imagens,", len(anns_train), "anotacoes")
    print("Validacao:", len(imgs_val), "imagens,", len(anns_val), "anotacoes")
    print("Pronto! Verifique as bboxes e retreine com: python mass_train.py")
