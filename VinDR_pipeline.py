"""
VinDR_pipeline.py - leitura de DICOM do VinDr-Mammo para o MMDetection 3.x

Registra:
  - LoadVindrDicom    : le o DICOM, aplica o windowing clinico, corrige
                        MONOCHROME1 e isola a mama
  - VindrMammoDataset : dataset COCO com uma classe ('mass')

Recorte da mama (opcional, LoadVindrDicom(recortar=True)):
  A imagem e recortada no retangulo que contem a mama antes do Resize, para
  que os pixels do Resize caiam em tecido e nao no fundo preto. A posicao do
  recorte vem do campo 'crop' das anotacoes geradas pelo
  VinDR_prepare_recorte.py, onde as caixas ja estao nas coordenadas do
  recorte. Sem esse campo (inferencia em imagem nova), o recorte e calculado
  na hora pela mesma funcao.

Diferencas em relacao ao dicom_pipeline.py (CBIS-DDSM):
  * O VinDr e mamografia digital (FFDM) "for presentation", sem as etiquetas
    de filme escaneado do CBIS, e o texto gravado nos cantos ja vem mascarado.
    Por isso NAO zeramos faixas fixas de borda: no FFDM a mama encosta na
    borda da imagem (lado da parede toracica) e as faixas cortariam tecido,
    e possivelmente massas. Basta manter a maior regiao conectada (a mama);
    marcadores de lateralidade, pequenos e separados, somem.
  * Imagens em MONOCHROME1 (fundo branco) sao invertidas.
  * Sem WindowCenter/WindowWidth, usa a VOI LUT do arquivo; sem nenhuma das
    duas, normaliza pelo minimo e maximo (como no CBIS).

O dicom_pipeline.py do CBIS nao e alterado: os treinos antigos continuam
reproduziveis.
"""
import numpy as np
import pydicom
import cv2
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS, DATASETS
from mmdet.datasets import CocoDataset

try:  # pydicom 3.x
    from pydicom.pixels import apply_voi_lut
except ImportError:  # pydicom 2.x
    from pydicom.pixel_data_handlers.util import apply_voi_lut


def _primeiro(valor):
    """WindowCenter/WindowWidth podem ter varios valores; usa o primeiro."""
    if isinstance(valor, pydicom.multival.MultiValue):
        valor = valor[0]
    return float(valor)


def _para_8bits(ds, usar_janela_dicom=True):
    """Pixels do DICOM -> uint8 (0-255), com windowing clinico."""
    try:
        bruto = ds.pixel_array
    except Exception as e:
        sintaxe = getattr(getattr(ds, 'file_meta', None), 'TransferSyntaxUID', '?')
        raise RuntimeError(
            f'Nao consegui decodificar os pixels (Transfer Syntax: {sintaxe}). '
            'Instale os decodificadores: '
            'pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg') from e

    px = bruto.astype(np.float32)

    # Modality LUT: a janela do DICOM se refere aos valores ja reescalados
    px = px * float(ds.get('RescaleSlope', 1) or 1) + float(ds.get('RescaleIntercept', 0) or 0)

    if usar_janela_dicom and 'WindowCenter' in ds and 'WindowWidth' in ds:
        wc, ww = _primeiro(ds.WindowCenter), _primeiro(ds.WindowWidth)
        lower, upper = wc - ww / 2, wc + ww / 2
        px = np.clip(px, lower, upper)
        px = (px - lower) / max(upper - lower, 1e-6) * 255.0
    else:
        if usar_janela_dicom and 'VOILUTSequence' in ds:
            px = apply_voi_lut(bruto, ds).astype(np.float32)
        pmin, pmax = float(px.min()), float(px.max())
        if pmax > pmin:
            px = (px - pmin) / (pmax - pmin) * 255.0

    img = np.clip(px, 0, 255).astype(np.uint8)

    # MONOCHROME1: valor alto = preto. Inverte para o padrao (fundo preto)
    if str(ds.get('PhotometricInterpretation', 'MONOCHROME2')).upper() == 'MONOCHROME1':
        img = 255 - img
    return img


def mascara_da_mama(gray):
    """Mascara (0/255) com a maior regiao conectada da imagem: a mama."""
    h, w = gray.shape
    k = max(5, int(min(h, w) * 0.005))  # ~0,5% do lado menor (~12 px no VinDr)

    _, th = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))  # tira ruido e texto

    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
    if n <= 1:  # nada encontrado: nao mascara nada
        return np.full_like(gray, 255)

    maior = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mascara = np.where(rotulos == maior, 255, 0).astype(np.uint8)

    # Fecha buracos internos e alarga um pouco para nao cortar a pele
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(gray)
    cv2.drawContours(mascara, contornos, -1, 255, -1)
    mascara = cv2.dilate(mascara, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    return mascara


def carregar_vindr(caminho, usar_janela_dicom=True, isolar_mama=True):
    """
    Le um DICOM do VinDr e devolve (imagem BGR uint8, mascara da mama).
    E a mesma funcao usada no treino (LoadVindrDicom) e na inferencia,
    para garantir o mesmo pre-processamento nos dois.
    """
    ds = pydicom.dcmread(caminho)
    gray = _para_8bits(ds, usar_janela_dicom)

    if isolar_mama:
        mascara = mascara_da_mama(gray)
        gray = cv2.bitwise_and(gray, mascara)
    else:
        mascara = np.full_like(gray, 255)

    img = np.stack([gray] * 3, axis=-1)  # 3 canais (BGR)
    return img, mascara


def retangulo_da_mama(mascara, margem=20):
    """(x0, y0, x1, y1) do menor retangulo que contem a mascara, com margem."""
    h, w = mascara.shape
    ys, xs = np.nonzero(mascara)
    if len(xs) == 0:
        return [0, 0, w, h]
    x0 = max(0, int(xs.min()) - margem)
    y0 = max(0, int(ys.min()) - margem)
    x1 = min(w, int(xs.max()) + 1 + margem)
    y1 = min(h, int(ys.max()) + 1 + margem)
    return [x0, y0, x1, y1]


def carregar_vindr_recortada(caminho, crop=None, usar_janela_dicom=True):
    """
    Le o DICOM, isola a mama e recorta no retangulo dela.
    Devolve (imagem recortada, mascara recortada, crop=[x0, y0, x1, y1]).
    Somar (x0, y0) leva uma coordenada do recorte para a imagem original.
    """
    img, mascara = carregar_vindr(caminho, usar_janela_dicom, isolar_mama=True)
    if crop is None:
        crop = retangulo_da_mama(mascara)
    x0, y0, x1, y1 = [int(v) for v in crop]
    return img[y0:y1, x0:x1].copy(), mascara[y0:y1, x0:x1].copy(), [x0, y0, x1, y1]


@TRANSFORMS.register_module()
class LoadVindrDicom(BaseTransform):
    """Carrega um DICOM do VinDr-Mammo no pipeline do MMDetection 3.x."""

    def __init__(self, usar_janela_dicom=True, isolar_mama=True, recortar=False):
        self.usar_janela_dicom = usar_janela_dicom
        self.isolar_mama = isolar_mama
        self.recortar = recortar

    def transform(self, results):
        crop = results.get('crop')
        if self.recortar:
            img, _, _ = carregar_vindr_recortada(results['img_path'], crop,
                                                 self.usar_janela_dicom)
        else:
            if crop is not None:
                raise ValueError('As anotacoes tem o campo "crop" (geradas pelo '
                                 'VinDR_prepare_recorte.py), mas o pipeline usa '
                                 'LoadVindrDicom sem recortar=True: as caixas nao '
                                 'bateriam com a imagem.')
            img, _ = carregar_vindr(results['img_path'],
                                    self.usar_janela_dicom, self.isolar_mama)
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results


@DATASETS.register_module()
class VindrMammoDataset(CocoDataset):
    """VinDr-Mammo em formato COCO (gerado pelo VinDR_prepare.py)."""
    METAINFO = {
        'classes': ('mass',),
        'palette': [(220, 20, 60)]
    }

    def parse_data_info(self, raw_data_info):
        # repassa ao pipeline a posicao do recorte, quando existir
        data_info = super().parse_data_info(raw_data_info)
        crop = raw_data_info['raw_img_info'].get('crop')
        if crop is not None:
            data_info['crop'] = crop
        return data_info
