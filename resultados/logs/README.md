# Logs de treinamento

Logs completos de todos os treinamentos do projeto, gerados pelo MMEngine. Cada execução tem:

- `<data_hora>.log` — log em texto: ambiente, config completo, perdas a cada 50 iterações e métricas COCO de validação a cada época;
- `<data_hora>/scalars.json` — os mesmos números em JSON, uma linha por registro (originalmente em `vis_data/`).

As pastas `teste/` têm a avaliação do melhor checkpoint no **conjunto de teste oficial** do VinDr-Mammo, feita pelo `VinDR_train.py` ao fim de cada treino.

Os caminhos da máquina local foram substituídos pelos caminhos relativos usados no repositório (`./`, `data/VinDr-Mammo`, `data/CBIS-DDSM/imagens_completas`). Nenhum log contém imagens ou dados dos exames.

Data e hora no nome da pasta: início da execução (horário de Brasília). mAP@0,50 e mAP referem-se à melhor época pelo mAP@0,50.

## CBIS-DDSM (validação: 247 imagens, 260 massas)

| Treino | Pasta | Configuração | Épocas | Melhor mAP@0,50 | mAP | Observação |
|---|---|---|---|---|---|---|
| — | `cbis_faster_rcnn/20260923_214224` | Faster R-CNN, 12 ép. | 0 | — | — | Interrompido antes da 1ª época (configuração do ambiente) |
| — | `cbis_faster_rcnn/20260923_214504` | Faster R-CNN, 12 ép. | 0 | — | — | Interrompido antes da 1ª época (configuração do ambiente) |
| A | `cbis_faster_rcnn/20260923_214600` | Faster R-CNN, 12 ép. | 12 | 94,1% (ép. 1) | 83,3% | **Inválido**: caixa = imagem inteira |
| B | `cbis_faster_rcnn/20261001_234527` | Faster R-CNN, 12 ép. | 12 | 54,9% (ép. 2) | 22,4% | **Inválido**: todas as caixas em (0, 0) (arquivo de recorte da ROI) |
| C | `cbis_faster_rcnn/20261006_225712` | B + remoção de artefatos | 12 | 55,6% (ép. 8) | 23,0% | **Inválido**: mesmas caixas de B |
| D | `cbis_faster_rcnn/20261007_225800` | Faster R-CNN, caixas corretas, 12 ép., LR ÷10 nas ép. 8 e 11 | 12 | 46,8% (ép. 10) | 20,0% | Primeiro resultado válido |
| E | `cbis_faster_rcnn/20261014_000140` | Repetição de D (outra semente) | 12 | 45,5% (ép. 9) | 19,5% | Reprodutibilidade de D |
| F | `cbis_faster_rcnn_40ep/20261014_180222` | 40 ép., cosseno, parada pelo mAP | 25 | 46,7% (ép. 9) | 17,6% | Parada antecipada na ép. 25 |
| G | `cbis_faster_rcnn_40ep_parada_loss/20261016_002551` | 40 ép., parada pela perda de validação | 10 | 44,5% (ép. 8) | 17,7% | Parou na ép. 10 guardando a ép. 2: a perda não serve como critério |
| — | `cbis_faster_rcnn_40ep_sem_parada/20261016_000104` | 40 ép., sem parada | 2 | 19,0% (ép. 1) | 5,3% | Interrompido na 2ª época |
| H | `cbis_faster_rcnn_40ep_sem_parada/20261016_174247` | 40 ép., sem parada, com perda de validação | 40 | 45,1% (ép. 7) | 18,0% | Overfitting nas últimas 20 épocas |
| I | `cbis_faster_rcnn_40ep_parada_map50/20261017_022145` | 40 ép., parada pelo **mAP@0,50** | 18 | **47,2%** (ép. 10) | 19,0% | Referência do CBIS-DDSM |
| Cascade | `cbis_cascade_rcnn_40ep_parada_map50/20261017_122353` | Cascade R-CNN, mesmo agendamento de I | 18 | 44,3% (ép. 10) | 18,8% | mAP@0,75 de 12,2% (I: 11,3%) |

## VinDr-Mammo (validação: 92 imagens, 97 massas · teste oficial: 219 imagens, 237 massas)

| Pasta | Configuração | Épocas | Validação mAP@0,50 | Teste mAP@0,50 | Teste mAP | Teste mAP@0,75 | Observação |
|---|---|---|---|---|---|---|---|
| `VinDR_faster_rcnn_40ep_parada_map50/20261028_232344` | Faster R-CNN | 1 | — | — | — | — | Interrompido: `WinError 1455` (memória virtual) |
| `VinDR_faster_rcnn_40ep_parada_map50/20261028_233314` | Faster R-CNN, 40 ép., parada pelo mAP@0,50 | 26 | 70,5% (ép. 18) | 58,7% | 27,7% | 24,6% | Referência do VinDr-Mammo |
| `VinDR_faster_rcnn_200ep_sem_parada/20261104_234134` | Faster R-CNN, 200 ép., sem parada | 200 | 69,2% (ép. 20) | 60,7% | 30,0% | 25,5% | Melhor época na 20ª; overfitting depois |
| `VinDR_faster_rcnn_recorte_40ep_parada_map50/20261105_231524` | Faster R-CNN com recorte da mama | 0 | — | — | — | — | Interrompido: `MemoryError` no carregamento |
| `VinDR_faster_rcnn_recorte_40ep_parada_map50/20261105_231743` | Faster R-CNN com recorte da mama (Resize 1600 × 1024) | 18 | 68,0% (ép. 10) | 56,8% | 26,0% | 21,2% | Massas médias: AP de 0,5% → 11,0% |
| `VinDR_fcos_40ep_parada_map50/20261107_015816` | FCOS | 27 | 69,0% (ép. 19) | 60,0% | 21,8% | 10,8% | |
| `VinDR_atss_40ep_parada_map50/20261107_033803` | ATSS | 15 | 71,3% (ép. 7) | 57,9% | 28,3% | 22,9% | |

O resultado do ensemble (Faster R-CNN + FCOS + ATSS) está em [`../resultado_test.json`](../resultado_test.json) e [`../resultado_val.json`](../resultado_val.json).
