"""Testes do metricas.tpr_fppi.  Rodar da raiz:  python3 -m unittest discover -s tests -v"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metricas import iou, tpr_fppi  # noqa: E402

MASSA = [100, 100, 200, 200]


def det(caixa, score):
    return list(caixa) + [score]


class TestTprFppi(unittest.TestCase):
    def test_uma_massa_uma_deteccao(self):
        self.assertEqual(tpr_fppi({1: [MASSA]}, {1: [det(MASSA, 0.9)]}, 0.3), (1.0, 0.0))

    def test_duas_deteccoes_na_mesma_massa(self):
        # O metodo antigo da CLI contava as duas como acerto (FPPI 0); aqui e 1 TP + 1 FP
        dets = [det(MASSA, 0.9), det([105, 105, 205, 205], 0.8)]
        self.assertEqual(tpr_fppi({1: [MASSA]}, {1: dets}, 0.3), (1.0, 1.0))

    def test_deteccao_sem_massa(self):
        self.assertEqual(tpr_fppi({1: [MASSA]}, {1: [det([500, 500, 600, 600], 0.9)]}, 0.3), (0.0, 1.0))

    def test_score_abaixo_do_limiar(self):
        self.assertEqual(tpr_fppi({1: [MASSA]}, {1: [det(MASSA, 0.29)]}, 0.3), (0.0, 0.0))

    def test_score_igual_ao_limiar_conta(self):
        self.assertEqual(tpr_fppi({1: [MASSA]}, {1: [det(MASSA, 0.3)]}, 0.3), (1.0, 0.0))

    def test_imagem_sem_massa(self):
        # imagem sem massa: entra no denominador do FPPI, e toda deteccao e FP
        gt = {1: [MASSA], 2: []}
        dets = {1: [det(MASSA, 0.9)], 2: [det([10, 10, 50, 50], 0.7)]}
        self.assertEqual(tpr_fppi(gt, dets, 0.3), (1.0, 0.5))

    def test_imagem_sem_massa_e_sem_deteccao(self):
        self.assertEqual(tpr_fppi({1: []}, {}, 0.3), (0.0, 0.0))

    def test_iou_abaixo_de_0_5_e_fp(self):
        deslocada = [150, 150, 250, 250]   # IoU = 2500 / 17500 ~ 0,14
        self.assertLess(iou(MASSA, deslocada), 0.5)
        self.assertEqual(tpr_fppi({1: [MASSA]}, {1: [det(deslocada, 0.9)]}, 0.3), (0.0, 1.0))

    def test_maior_score_fica_com_a_massa(self):
        # a deteccao de maior score e pareada primeiro, mesmo com IoU menor
        boa = det([100, 100, 200, 200], 0.6)
        pior = det([110, 110, 210, 210], 0.9)
        gt = {1: [MASSA]}
        self.assertEqual(tpr_fppi(gt, {1: [boa, pior]}, 0.3), (1.0, 1.0))


def _tpr_fppi_original(gt, det, thr, iou_min=0.5):
    """Copia literal de VinDR_ensemble.py:134-161 no commit b5f534d (antes da mudanca)."""
    def iou_orig(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / uniao if uniao > 0 else 0.0

    achadas = total = fp = 0
    for iid, caixas_gt in gt.items():
        dets = sorted([d for d in det.get(iid, []) if d[4] >= thr], key=lambda d: -d[4])
        usadas = set()
        for d in dets:
            melhor, k_melhor = 0.0, -1
            for k, g in enumerate(caixas_gt):
                if k not in usadas:
                    v = iou_orig(d, g)
                    if v > melhor:
                        melhor, k_melhor = v, k
            if melhor >= iou_min:
                usadas.add(k_melhor)
            else:
                fp += 1
        achadas += len(usadas)
        total += len(caixas_gt)
    return achadas / max(total, 1), fp / max(len(gt), 1)


class TestIgualAoEnsembleOriginal(unittest.TestCase):
    def test_dados_aleatorios(self):
        rng = random.Random(0)

        def caixa():
            x, y = rng.uniform(0, 900), rng.uniform(0, 900)
            return [x, y, x + rng.uniform(20, 150), y + rng.uniform(20, 150)]

        for _ in range(300):
            gt, dets = {}, {}
            for iid in range(rng.randint(1, 15)):
                gt[iid] = [caixa() for _ in range(rng.randint(0, 3))]
                # deteccoes perto das massas (para gerar acertos) e soltas
                perto = [[v + rng.uniform(-15, 15) for v in g] for g in gt[iid] for _ in range(rng.randint(0, 2))]
                dets[iid] = [b + [round(rng.random(), 2)] for b in perto + [caixa() for _ in range(rng.randint(0, 4))]]
            for thr in (0.0, 0.3, 0.5):
                self.assertEqual(tpr_fppi(gt, dets, thr), _tpr_fppi_original(gt, dets, thr))


if __name__ == '__main__':
    unittest.main()
