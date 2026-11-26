# Falsos positivos em exames normais (teste oficial do VinDr-Mammo)

Imagens: 219 com massa (237 massas) + 1000 normais ("No Finding") sorteadas (semente 42) de 3643 normais do split test. Mesmas imagens para todos os modelos.
Deteccoes sem filtros. TPR com IoU >= 0,5 e pareamento 1 para 1 (metricas.py). Gerado por avaliar_fp_normais.py; numeros agregados, nenhum ID.

## Limiar de score 0,30

| Modelo | Checkpoint | TPR (massas) | FPPI nas imagens com massa | FPPI em normais | Normais com >= 1 caixa |
|---|---|---|---|---|---|
| r50aug_s0 | best_coco_bbox_mAP_50_epoch_20.pth | 73,0% | 1,07 | 0,97 | 58,9% |
| r50aug_s1 | best_coco_bbox_mAP_50_epoch_15.pth | 70,5% | 0,78 | 0,61 | 44,0% |
| r50aug_s2 | best_coco_bbox_mAP_50_epoch_25.pth | 70,9% | 0,97 | 0,83 | 55,3% |
| r50augneg_s0 | best_coco_bbox_mAP_50_epoch_29.pth | 64,1% | 0,56 | 0,38 | 31,9% |
| r50augneg_s1 | best_coco_bbox_mAP_50_epoch_18.pth | 67,5% | 0,88 | 0,61 | 44,4% |
| r50augneg_s2 | best_coco_bbox_mAP_50_epoch_17.pth | 71,7% | 1,04 | 0,74 | 49,7% |

## Limiar de score 0,50

| Modelo | Checkpoint | TPR (massas) | FPPI nas imagens com massa | FPPI em normais | Normais com >= 1 caixa |
|---|---|---|---|---|---|
| r50aug_s0 | best_coco_bbox_mAP_50_epoch_20.pth | 68,4% | 0,66 | 0,59 | 42,6% |
| r50aug_s1 | best_coco_bbox_mAP_50_epoch_15.pth | 63,7% | 0,32 | 0,28 | 24,4% |
| r50aug_s2 | best_coco_bbox_mAP_50_epoch_25.pth | 67,9% | 0,59 | 0,50 | 39,5% |
| r50augneg_s0 | best_coco_bbox_mAP_50_epoch_29.pth | 61,2% | 0,39 | 0,24 | 22,3% |
| r50augneg_s1 | best_coco_bbox_mAP_50_epoch_18.pth | 61,6% | 0,50 | 0,34 | 28,5% |
| r50augneg_s2 | best_coco_bbox_mAP_50_epoch_17.pth | 65,4% | 0,51 | 0,34 | 29,4% |

## Limiar de score 0,70

| Modelo | Checkpoint | TPR (massas) | FPPI nas imagens com massa | FPPI em normais | Normais com >= 1 caixa |
|---|---|---|---|---|---|
| r50aug_s0 | best_coco_bbox_mAP_50_epoch_20.pth | 62,9% | 0,37 | 0,31 | 26,8% |
| r50aug_s1 | best_coco_bbox_mAP_50_epoch_15.pth | 49,8% | 0,14 | 0,10 | 9,2% |
| r50aug_s2 | best_coco_bbox_mAP_50_epoch_25.pth | 64,1% | 0,34 | 0,29 | 26,1% |
| r50augneg_s0 | best_coco_bbox_mAP_50_epoch_29.pth | 56,5% | 0,22 | 0,16 | 15,0% |
| r50augneg_s1 | best_coco_bbox_mAP_50_epoch_18.pth | 55,3% | 0,25 | 0,15 | 14,3% |
| r50augneg_s2 | best_coco_bbox_mAP_50_epoch_17.pth | 55,7% | 0,22 | 0,15 | 14,3% |

## TPR com FPPI em normais fixo (varrendo o limiar)

| Modelo | TPR @ FPPI 0,25 | TPR @ FPPI 0,50 | TPR @ FPPI 1,00 |
|---|---|---|---|
| r50aug_s0 | 61,2% | 67,1% | 73,4% |
| r50aug_s1 | 61,6% | 69,2% | 72,2% |
| r50aug_s2 | 62,9% | 67,9% | 72,2% |
| r50augneg_s0 | 61,2% | 66,2% | 71,7% |
| r50augneg_s1 | 59,1% | 65,8% | 72,6% |
| r50augneg_s2 | 60,3% | 67,9% | 73,8% |

## Media ± desvio padrao por configuracao (limiar 0,3) e teste t de Welch

| Metrica | r50aug (n=3) | r50augneg (n=3) | p (Welch, 1a x 2a) |
|---|---|---|---|
| TPR | 71,4% ± 1,4% | 67,8% ± 3,8% | 0,232 |
| FPPI em normais | 0,80 ± 0,19 | 0,58 ± 0,19 | 0,210 |
| Normais com caixa | 52,7% ± 7,8% | 42,0% ± 9,1% | 0,198 |
| TPR @ FPPI 0,5 | 68,1% ± 1,1% | 66,7% ± 1,1% | 0,189 |
| TPR @ FPPI 1,0 | 72,6% ± 0,7% | 72,7% ± 1,1% | 0,860 |

