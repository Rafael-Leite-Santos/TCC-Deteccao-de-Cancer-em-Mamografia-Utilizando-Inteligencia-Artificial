# Protocolo de avaliacao do Digital Eye aplicado a `VinDR_faster_rcnn_40ep_parada_map50`

Protocolo (lido no codigo do Digital Eye em 17/11/2026): NMS IoU 0,1 -> score > 0,2 -> AP VOC (area) com IoU 0,1.
Gerado por avaliar_protocolo_digitaleye.py a partir do cache do analise_erros.py. Numeros agregados.
Os numeros do Digital Eye (KETEM, privado) continuam NAO comparaveis: o dataset e outro.

## val (92 imagens, 97 massas)

| Variante | AP (IoU 0,1) | AP (IoU 0,5) | TPR / FPPI (IoU 0,1, score > 0,2) | TPR / FPPI (IoU 0,5, score > 0,2) |
|---|---|---|---|---|
| Protocolo Digital Eye (NMS 0,1 + score > 0,2) | 76.0% | 66.7% | 86.6% / 1.08 | 77.3% / 1.17 |
| Sem NMS 0,1 / limiar (saida do nosso modelo, score >= 0,05) | 77.3% | 70.6% | 87.6% / 1.49 | 81.4% / 1.55 |

## test (219 imagens, 237 massas)

| Variante | AP (IoU 0,1) | AP (IoU 0,5) | TPR / FPPI (IoU 0,1, score > 0,2) | TPR / FPPI (IoU 0,5, score > 0,2) |
|---|---|---|---|---|
| Protocolo Digital Eye (NMS 0,1 + score > 0,2) | 65.7% | 55.9% | 82.7% / 1.43 | 70.9% / 1.56 |
| Sem NMS 0,1 / limiar (saida do nosso modelo, score >= 0,05) | 67.1% | 58.8% | 82.7% / 1.96 | 73.0% / 2.06 |

