# Backbones no Faster R-CNN (ponto 4 do orientador, 16/11/2026)

Mesmo config da 2a rodada da referencia (`configs/VinDR_faster_rcnn_wd1e-4.py`: ResNet-50, weight decay 1e-4,
ValLossHook), mudando so o backbone (depth, pre-treino ImageNet do torchvision, canais do FPN).
40 epocas, cosseno, early stopping no mAP@0,50 (paciencia 8). Uma rodada cada, sem IC.
Fonte: logs de treino (`work_dirs/VinDR_faster_rcnn_r18|r34_40ep_parada_map50/<data>/<data>.log`) e de teste (`.../teste/`).

| Backbone | Val mAP@50 (ep.) | Val mAP | Val mAP@75 | Teste mAP@50 | Teste mAP | Teste mAP@75 | Parou na ep. |
|---|---|---|---|---|---|---|---|
| ResNet-18 (20261116_152840) | 73,2% (13) | 37,2% | 35,2% | 62,6% | 30,0% | 23,7% | 21 |
| ResNet-34 (20261116_165909) | 70,5% (11) | 35,0% | 29,0% | 62,8% | 30,6% | 27,4% | 19 |
| ResNet-50, 2a rodada da ref. (20261116_013427) | 69,6% (7) | 34,2% | 27,1% | 60,6% | 29,7% | 24,5% | 15 |
| ResNet-50, referencia original (20261028_233314) | 70,5% (18) | 36,0% | 31,0% | 58,7% | 27,7% | 24,6% | 26 |

Diferenca loss de validacao - loss de treino na epoca 15: ResNet-18 = 0,033; ResNet-34 = 0,072; ResNet-50 = 0,095.
Ruido de referencia: duas rodadas identicas da ResNet-50 diferiram ~1 pt (val) e ~2 pt (teste) de mAP@0,50.
O aviso "unexpected key in source state_dict: fc.weight, fc.bias" e normal (camada de classificacao do ImageNet, nao usada).

Grafico: `curvas_backbones.png`.
