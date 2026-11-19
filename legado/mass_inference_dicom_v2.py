"""
mass_inference_dicom_v2.py - deteccao de massas em um DICOM do CBIS-DDSM

Uso:
  python mass_inference_dicom_v2.py imagem.dcm                       (melhor checkpoint do config)
  python mass_inference_dicom_v2.py imagem.dcm modelo.pth            (outro checkpoint)
  python mass_inference_dicom_v2.py imagem.dcm modelo.pth config.py  (outro modelo, ex.: Cascade R-CNN)

Salva <nome>_roi_deteccao.png com as deteccoes. Descarta deteccoes com centro
fora da mama e caixas maiores que 30% da imagem.
"""
import glob
import os
import sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pydicom
import cv2
import torch

from mmengine.config import Config
from mmdet.registry import DATASETS
from mmdet.datasets import CocoDataset
from mmdet.apis import init_detector, inference_detector

import dicom_pipeline
import mmdet.models
import mmdet.datasets
import mmdet.evaluation


@DATASETS.register_module(force=True)
class CbisDdsmDataset(CocoDataset):
    METAINFO = {
        "classes": ("mass",),
        "palette": [(220, 20, 60)]
    }


def carregar_dicom(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    pixels = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "WindowCenter"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if isinstance(wc, pydicom.multival.MultiValue):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
        lower = wc - ww / 2
        upper = wc + ww / 2
        pixels = np.clip(pixels, lower, upper)
        pixels = (pixels - lower) / (upper - lower) * 255.0
    else:
        pmin, pmax = pixels.min(), pixels.max()
        pixels = (pixels - pmin) / (pmax - pmin + 1e-8) * 255.0
    pixels = pixels.astype(np.uint8)
    if pixels.ndim == 2:
        pixels = np.stack([pixels] * 3, axis=-1)
    return pixels


def remover_artefatos_borda(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    margem_topo    = int(h * 0.12)
    margem_base    = int(h * 0.04)
    margem_direita = int(w * 0.15)
    margem_esq     = int(w * 0.04)

    mascara_borda = np.ones((h, w), dtype=np.uint8) * 255
    mascara_borda[:margem_topo, :]       = 0
    mascara_borda[h-margem_base:, :]    = 0
    mascara_borda[:, w-margem_direita:] = 0
    mascara_borda[:, :margem_esq]       = 0

    gray_sem_borda = cv2.bitwise_and(gray, mascara_borda)

    _, thresh = cv2.threshold(gray_sem_borda, 10, 255, cv2.THRESH_BINARY)

    kernel = np.ones((40, 40), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        return img, (0, 0, img.shape[1], img.shape[0])

    maior = max(contornos, key=cv2.contourArea)
    mascara_final = np.zeros_like(gray)
    cv2.drawContours(mascara_final, [maior], -1, 255, -1)

    img_limpa = img.copy()
    img_limpa[mascara_final == 0] = 0

    x, y, ww, hh = cv2.boundingRect(maior)
    return img_limpa, (x, y, ww, hh)


CONFIG_PADRAO = "configs/cbis_faster_rcnn.py"


def melhor_checkpoint(config_path):
    work_dir = Config.fromfile(config_path).work_dir
    candidatos = glob.glob(os.path.join(work_dir, "best_*.pth"))
    if not candidatos:
        sys.exit("Nenhum best_*.pth em " + work_dir + ": informe o checkpoint.")
    return max(candidatos, key=os.path.getmtime)


def inferencia_melhorada(dicom_path, checkpoint_path, score_thr=0.3, config_path=CONFIG_PADRAO):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Carregando modelo de:", checkpoint_path)
    model = init_detector(config_path, checkpoint_path, device=device)

    print("Carregando imagem DICOM:", dicom_path)
    img_original = carregar_dicom(dicom_path)
    
    print("Removendo artefatos de borda...")
    img_crop, (crop_x, crop_y, crop_w, crop_h) = remover_artefatos_borda(img_original)
    print("Regiao da mama isolada:", img_crop.shape[:2])

    print("Executando inferencia...")
    resultado = inference_detector(model, img_crop)

    pred_instances = resultado.pred_instances
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()

    # Filtrar por score
    mask = scores >= score_thr
    bboxes = bboxes[mask]
    scores = scores[mask]

    print("Deteccoes encontradas (score >=", score_thr, "):", len(bboxes))

    # Desenhar na imagem original (convertendo coordenadas de volta)
    img_resultado = img_original.copy()
    
        # Criar mascara da mama para filtrar deteccoes fora dela
    gray_orig = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    _, thresh_orig = cv2.threshold(gray_orig, 10, 255, cv2.THRESH_BINARY)
    kernel_orig = np.ones((40, 40), np.uint8)
    thresh_orig = cv2.morphologyEx(thresh_orig, cv2.MORPH_CLOSE, kernel_orig)
    contornos_orig, _ = cv2.findContours(thresh_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_mama = np.zeros(img_original.shape[:2], dtype=np.uint8)
    if contornos_orig:
        maior_orig = max(contornos_orig, key=cv2.contourArea)
        cv2.drawContours(mascara_mama, [maior_orig], -1, 255, -1)

    for i, (bbox, score) in enumerate(zip(bboxes, scores)):
        x1 = int(bbox[0]) + crop_x
        y1 = int(bbox[1]) + crop_y
        x2 = int(bbox[2]) + crop_x
        y2 = int(bbox[3]) + crop_y

        # Calcular centro da bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Verificar se o centro esta dentro da mama
        cy_clamped = min(cy, mascara_mama.shape[0]-1)
        cx_clamped = min(cx, mascara_mama.shape[1]-1)
        if mascara_mama[cy_clamped, cx_clamped] == 0:
            print("  [{}] Ignorada (fora da mama) | Score: {:.3f}".format(i+1, score))
            continue

        area_bbox = (x2 - x1) * (y2 - y1)
        area_img = img_original.shape[0] * img_original.shape[1]
        if area_bbox > area_img * 0.30:
            print("  [{}] Ignorada (bbox muito grande) | Score: {:.3f}".format(i+1, score))
            continue

    # Salvar resultado
    nome_saida = os.path.splitext(os.path.basename(dicom_path))[0] + "_roi_deteccao.png"
    cv2.imwrite(nome_saida, img_resultado)
    print("Imagem salva:", nome_saida)
    return bboxes, scores


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python mass_inference_dicom_v2.py caminho/para/imagem.dcm")
        sys.exit(1)

    config = sys.argv[3] if len(sys.argv) > 3 else CONFIG_PADRAO
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else melhor_checkpoint(config)
    inferencia_melhorada(
        dicom_path=sys.argv[1],
        checkpoint_path=checkpoint,
        score_thr=0.3,
        config_path=config
    )
