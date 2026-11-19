# Metricas em funcao do IoU minimo de acerto (ensemble v3, 19/11/2026)

Gerado com `python reavaliar_iou.py --split val` e `python reavaliar_iou.py` sobre o cache do
`VinDR_ensemble.py` (Faster `sementes/r50aug_s2` ep. 25, FCOS `sementes/fcosaug_s0` ep. 23,
ATSS `sementes/atssaug_s0` ep. 23; WBF pesos [1,1,1], IoU 0,55, skip 0,05). AP no padrao COCO
(area all, ate 100 deteccoes); TPR/FPPI com score >= 0,30 (`metricas.tpr_fppi`). Uma execucao, sem IC.
IoU 0,5 = criterio do TCC; IoU 0,1 = limiar de acerto do Digital Eye (o protocolo completo dele esta em
`analises/protocolo_digitaleye/`). Numeros do Digital Eye (KETEM) continuam nao comparaveis.

## Teste oficial (219 imagens, 237 massas): AP / TPR / FPPI

| Modelo | IoU 0,1 | IoU 0,2 | IoU 0,3 | IoU 0,4 | IoU 0,5 |
|---|---|---|---|---|---|
| Faster R-CNN | 71,8 / 78,1 / 0,89 | 71,5 / 77,6 / 0,90 | 70,0 / 75,1 / 0,93 | 68,6 / 73,4 / 0,95 | 65,0 / 70,9 / 0,97 |
| FCOS | 76,9 / 59,9 / 0,13 | 76,0 / 59,5 / 0,13 | 73,7 / 58,6 / 0,14 | 70,9 / 57,8 / 0,15 | 69,0 / 57,4 / 0,16 |
| ATSS | 72,0 / 70,5 / 0,34 | 71,2 / 69,6 / 0,35 | 68,7 / 68,4 / 0,36 | 67,3 / 67,5 / 0,37 | 65,0 / 65,0 / 0,40 |
| Ensemble WBF | 73,1 / 73,8 / 0,42 | 72,6 / 73,4 / 0,43 | 70,0 / 72,2 / 0,44 | 68,7 / 70,9 / 0,46 | 66,4 / 69,6 / 0,47 |

## Validacao (92 imagens, 97 massas): AP / TPR / FPPI

| Modelo | IoU 0,1 | IoU 0,2 | IoU 0,3 | IoU 0,4 | IoU 0,5 |
|---|---|---|---|---|---|
| Faster R-CNN | 81,4 / 85,6 / 0,61 | 81,4 / 85,6 / 0,61 | 81,4 / 85,6 / 0,61 | 79,9 / 83,5 / 0,63 | 77,2 / 80,4 / 0,66 |
| FCOS | 85,9 / 66,0 / 0,11 | 85,6 / 66,0 / 0,11 | 84,2 / 66,0 / 0,11 | 81,3 / 63,9 / 0,13 | 79,3 / 62,9 / 0,14 |
| ATSS | 81,4 / 74,2 / 0,22 | 81,2 / 74,2 / 0,22 | 80,0 / 73,2 / 0,23 | 78,5 / 72,2 / 0,24 | 76,2 / 71,1 / 0,25 |
| Ensemble WBF | 83,7 / 76,3 / 0,25 | 83,6 / 76,3 / 0,25 | 82,9 / 76,3 / 0,25 | 81,0 / 74,2 / 0,27 | 78,9 / 73,2 / 0,28 |

Conferencia: as colunas IoU 0,5 batem com `resultados_ensemble/resultado_{val,test}.json` (ensemble v3).
Grafico: `metricas_por_iou.png` (teste).
