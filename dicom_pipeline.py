import numpy as np
import pydicom
import cv2
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadDicomImage(BaseTransform):
    """
    Carrega uma imagem DICOM com windowing clínico e remoção de artefatos.
    """

    def __init__(self, use_dicom_window=True, remove_artifacts=True):
        self.use_dicom_window = use_dicom_window
        self.remove_artifacts = remove_artifacts

    def _aplicar_windowing(self, pixels, ds):
        if self.use_dicom_window and hasattr(ds, 'WindowCenter'):
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
            if pmax > pmin:
                pixels = (pixels - pmin) / (pmax - pmin) * 255.0
        return pixels.astype(np.uint8)

    def _remover_artefatos(self, img):
        """
        Remove artefatos zerando faixas de borda fixas e
        isolando a maior região conectada da mama.
        """
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Zerar bordas fixas (onde os artefatos sempre aparecem)
        margem_topo    = int(h * 0.12)   # 12% do topo
        margem_base    = int(h * 0.04)   # 4% da base
        margem_direita = int(w * 0.14)   # 14% da direita
        margem_esq     = int(w * 0.04)   # 4% da esquerda

        mascara_borda = np.ones((h, w), dtype=np.uint8) * 255
        mascara_borda[:margem_topo, :]       = 0  # zera topo
        mascara_borda[h-margem_base:, :]    = 0  # zera base
        mascara_borda[:, w-margem_direita:] = 0  # zera direita
        mascara_borda[:, :margem_esq]       = 0  # zera esquerda

        # 2. Aplicar mascara de borda
        gray_sem_borda = cv2.bitwise_and(gray, mascara_borda)

        # 3. Threshold para encontrar a mama
        _, thresh = cv2.threshold(gray_sem_borda, 10, 255, cv2.THRESH_BINARY)

        # 4. Morfologia para fechar buracos internos
        kernel = np.ones((40, 40), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. Pegar o maior contorno (a mama)
        contornos, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return img

        maior = max(contornos, key=cv2.contourArea)
        mascara_final = np.zeros_like(gray)
        cv2.drawContours(mascara_final, [maior], -1, 255, -1)

        # 6. Aplicar mascara na imagem original
        img_limpa = img.copy()
        img_limpa[mascara_final == 0] = 0

        return img_limpa

    def transform(self, results):
        filepath = results['img_path']

        # 1. Ler DICOM
        ds = pydicom.dcmread(filepath)
        pixels = ds.pixel_array.astype(np.float32)

        # 2. Aplicar windowing
        pixels = self._aplicar_windowing(pixels, ds)

        # 3. Converter para BGR (3 canais)
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)

        # 4. Remover artefatos de borda
        if self.remove_artifacts:
            pixels = self._remover_artefatos(pixels)

        # 5. Preencher resultados
        results['img'] = pixels
        results['img_shape'] = pixels.shape[:2]
        results['ori_shape'] = pixels.shape[:2]

        return results