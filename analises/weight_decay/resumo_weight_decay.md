# Weight decay no Faster R-CNN (ponto 2 do orientador, 15-16/11/2026)

Mesmo config da referencia (`configs/VinDR_faster_rcnn.py`) mudando so `weight_decay`, com
`ValLossHook`. 40 epocas, cosseno, early stopping no mAP@0,50 (paciencia 8). Uma rodada cada, sem IC.
Fonte: logs de treino (`work_dirs/VinDR_faster_rcnn_wd*_40ep_parada_map50/<data>/<data>.log`) e de teste (`.../teste/`).

| weight_decay | Val mAP@50 (ep.) | Val mAP | Val mAP@75 | Teste mAP@50 | Teste mAP | Teste mAP@75 | Parou na ep. |
|---|---|---|---|---|---|---|---|
| 1e-4, referencia original (20261028_233314) | 70,5% (18) | 36,0% | 31,0% | 58,7% | 27,7% | 24,6% | 26 |
| 1e-4, 2a rodada (20261116_013427) | 69,6% (7) | 34,2% | 27,1% | 60,6% | 29,7% | 24,5% | 15 |
| 5e-4 (20261115_231046) | 72,5% (10) | 35,8% | 30,1% | 60,8% | 27,1% | 21,0% | 18 |
| 1e-3 (20261116_002918) | 69,8% (7) | 33,8% | 31,0% | 60,8% | 29,4% | 23,7% | 15 |

Diferenca loss de validacao - loss de treino (treino = media das iteracoes registradas na epoca):
- epoca 15: 1e-4 = 0,095; 5e-4 = 0,065; 1e-3 = 0,063
- a loss de validacao fica estavel em ~0,42-0,43 nos tres; a diferenca cresce porque a loss de treino cai a partir da epoca ~10.

Grafico: `curvas_weight_decay.png` (loss, diferenca e mAP@0,50 de validacao por epoca).
