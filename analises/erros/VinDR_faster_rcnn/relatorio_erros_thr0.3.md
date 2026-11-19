# Analise de erros: VinDR_faster_rcnn

Checkpoint: `./work_dirs/VinDR_faster_rcnn_40ep_parada_map50/best_coco_bbox_mAP_50_epoch_18.pth`. Deteccoes sem filtros, como na avaliacao do treino.
Gerado por analise_erros.py. Numeros agregados; nenhum ID ou dado de paciente.

# Conjunto: val (92 imagens, 97 massas, limiar de score 0.3)

- TPR/FPPI com IoU >= 0.5: TPR 75.3%, FPPI 1.01
- TPR/FPPI com IoU >= 0.1: TPR 81.4%, FPPI 0.95
  (IoU 0,1 aqui e so o limiar de pareamento; NAO e ainda o protocolo do Digital Eye)

## Massas anotadas por tipo

| tipo | n (%) |
|---|---|
| acerto | 73 (75.3%) |
| erro_localizacao | 6 (6.2%) |
| perdida_limiar | 10 (10.3%) |
| perdida | 8 (8.2%) |

## Massas por faixa de tamanho (lado da caixa anotada)

| faixa | n | acerto | erro_localizacao | perdida_limiar | perdida |
|---|---|---|---|---|---|
| 0-150 px | 16 | 6 (37.5%) | 1 (6.2%) | 6 (37.5%) | 3 (18.8%) |
| 150-250 px | 27 | 19 (70.4%) | 3 (11.1%) | 1 (3.7%) | 4 (14.8%) |
| 250-350 px | 31 | 27 (87.1%) | 2 (6.5%) | 2 (6.5%) | 0 (0.0%) |
| >=350 px | 23 | 21 (91.3%) | 0 (0.0%) | 1 (4.3%) | 1 (4.3%) |

## Falsos positivos (score >= 0.3): 93 no total, 1.01 por imagem

| tipo | n (%) |
|---|---|
| fp_duplicata | 4 (4.3%) |
| fp_localizacao | 17 (18.3%) |
| fp_fundo | 72 (77.4%) |

## Tamanho e posicao das caixas previstas

- acerto (n=73): IoU mediano 0.76; razao de area prevista/anotada mediana 0.96 (Q1-Q3 0.83-1.16; 29 maiores, 44 menores que a anotacao); deslocamento do centro mediano 0.07 x lado da massa
- erro_localizacao (n=6): IoU mediano 0.41; razao de area prevista/anotada mediana 1.32 (Q1-Q3 0.49-2.42; 3 maiores, 3 menores que a anotacao); deslocamento do centro mediano 0.21 x lado da massa
- perdidas pelo limiar: melhor score com IoU >= 0,5 mediano 0.23 (min 0.05, max 0.29)

# Conjunto: test (219 imagens, 237 massas, limiar de score 0.3)

- TPR/FPPI com IoU >= 0.5: TPR 70.0%, FPPI 1.44
- TPR/FPPI com IoU >= 0.1: TPR 78.5%, FPPI 1.35
  (IoU 0,1 aqui e so o limiar de pareamento; NAO e ainda o protocolo do Digital Eye)

## Massas anotadas por tipo

| tipo | n (%) |
|---|---|
| acerto | 166 (70.0%) |
| erro_localizacao | 20 (8.4%) |
| perdida_limiar | 22 (9.3%) |
| perdida | 29 (12.2%) |

## Massas por faixa de tamanho (lado da caixa anotada)

| faixa | n | acerto | erro_localizacao | perdida_limiar | perdida |
|---|---|---|---|---|---|
| 0-150 px | 54 | 24 (44.4%) | 2 (3.7%) | 14 (25.9%) | 14 (25.9%) |
| 150-250 px | 89 | 64 (71.9%) | 7 (7.9%) | 7 (7.9%) | 11 (12.4%) |
| 250-350 px | 49 | 38 (77.6%) | 6 (12.2%) | 1 (2.0%) | 4 (8.2%) |
| >=350 px | 45 | 40 (88.9%) | 5 (11.1%) | 0 (0.0%) | 0 (0.0%) |

## Falsos positivos (score >= 0.3): 316 no total, 1.44 por imagem

| tipo | n (%) |
|---|---|
| fp_duplicata | 4 (1.3%) |
| fp_localizacao | 48 (15.2%) |
| fp_fundo | 264 (83.5%) |

## Tamanho e posicao das caixas previstas

- acerto (n=166): IoU mediano 0.77; razao de area prevista/anotada mediana 0.99 (Q1-Q3 0.80-1.14; 79 maiores, 87 menores que a anotacao); deslocamento do centro mediano 0.06 x lado da massa
- erro_localizacao (n=20): IoU mediano 0.36; razao de area prevista/anotada mediana 0.45 (Q1-Q3 0.33-1.68; 7 maiores, 13 menores que a anotacao); deslocamento do centro mediano 0.23 x lado da massa
- perdidas pelo limiar: melhor score com IoU >= 0,5 mediano 0.15 (min 0.05, max 0.28)
