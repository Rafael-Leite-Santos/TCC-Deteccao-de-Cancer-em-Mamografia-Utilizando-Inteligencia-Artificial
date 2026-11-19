# Analise de erros: r50aug_s2

Checkpoint: `./work_dirs/sementes/r50aug_s2/best_coco_bbox_mAP_50_epoch_25.pth`. Deteccoes sem filtros, como na avaliacao do treino.
Gerado por analise_erros.py. Numeros agregados; nenhum ID ou dado de paciente.

# Conjunto: val (92 imagens, 97 massas, limiar de score 0.3)

- TPR/FPPI com IoU >= 0.5: TPR 80.4%, FPPI 0.66
- TPR/FPPI com IoU >= 0.1: TPR 85.6%, FPPI 0.61
  (IoU 0,1 aqui e so o limiar de pareamento; NAO e ainda o protocolo do Digital Eye)

## Massas anotadas por tipo

| tipo | n (%) |
|---|---|
| acerto | 78 (80.4%) |
| erro_localizacao | 5 (5.2%) |
| perdida_limiar | 4 (4.1%) |
| perdida | 10 (10.3%) |

## Massas por faixa de tamanho (lado da caixa anotada)

| faixa | n | acerto | erro_localizacao | perdida_limiar | perdida |
|---|---|---|---|---|---|
| 0-150 px | 16 | 5 (31.2%) | 1 (6.2%) | 3 (18.8%) | 7 (43.8%) |
| 150-250 px | 27 | 21 (77.8%) | 3 (11.1%) | 1 (3.7%) | 2 (7.4%) |
| 250-350 px | 31 | 29 (93.5%) | 1 (3.2%) | 0 (0.0%) | 1 (3.2%) |
| >=350 px | 23 | 23 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

## Falsos positivos (score >= 0.3): 61 no total, 0.66 por imagem

| tipo | n (%) |
|---|---|
| fp_duplicata | 5 (8.2%) |
| fp_localizacao | 8 (13.1%) |
| fp_fundo | 48 (78.7%) |

## Tamanho e posicao das caixas previstas

- acerto (n=78): IoU mediano 0.77; razao de area prevista/anotada mediana 0.98 (Q1-Q3 0.81-1.13; 37 maiores, 41 menores que a anotacao); deslocamento do centro mediano 0.06 x lado da massa
- erro_localizacao (n=5): IoU mediano 0.40; razao de area prevista/anotada mediana 2.32 (Q1-Q3 0.92-2.44; 3 maiores, 2 menores que a anotacao); deslocamento do centro mediano 0.22 x lado da massa
- perdidas pelo limiar: melhor score com IoU >= 0,5 mediano 0.14 (min 0.09, max 0.23)

# Conjunto: test (219 imagens, 237 massas, limiar de score 0.3)

- TPR/FPPI com IoU >= 0.5: TPR 70.9%, FPPI 0.97
- TPR/FPPI com IoU >= 0.1: TPR 78.1%, FPPI 0.89
  (IoU 0,1 aqui e so o limiar de pareamento; NAO e ainda o protocolo do Digital Eye)

## Massas anotadas por tipo

| tipo | n (%) |
|---|---|
| acerto | 168 (70.9%) |
| erro_localizacao | 17 (7.2%) |
| perdida_limiar | 20 (8.4%) |
| perdida | 32 (13.5%) |

## Massas por faixa de tamanho (lado da caixa anotada)

| faixa | n | acerto | erro_localizacao | perdida_limiar | perdida |
|---|---|---|---|---|---|
| 0-150 px | 54 | 24 (44.4%) | 1 (1.9%) | 14 (25.9%) | 15 (27.8%) |
| 150-250 px | 89 | 67 (75.3%) | 4 (4.5%) | 5 (5.6%) | 13 (14.6%) |
| 250-350 px | 49 | 37 (75.5%) | 7 (14.3%) | 1 (2.0%) | 4 (8.2%) |
| >=350 px | 45 | 40 (88.9%) | 5 (11.1%) | 0 (0.0%) | 0 (0.0%) |

## Falsos positivos (score >= 0.3): 213 no total, 0.97 por imagem

| tipo | n (%) |
|---|---|
| fp_duplicata | 3 (1.4%) |
| fp_localizacao | 32 (15.0%) |
| fp_fundo | 178 (83.6%) |

## Tamanho e posicao das caixas previstas

- acerto (n=168): IoU mediano 0.78; razao de area prevista/anotada mediana 0.98 (Q1-Q3 0.79-1.14; 77 maiores, 91 menores que a anotacao); deslocamento do centro mediano 0.06 x lado da massa
- erro_localizacao (n=17): IoU mediano 0.31; razao de area prevista/anotada mediana 0.40 (Q1-Q3 0.28-2.14; 6 maiores, 11 menores que a anotacao); deslocamento do centro mediano 0.27 x lado da massa
- perdidas pelo limiar: melhor score com IoU >= 0,5 mediano 0.11 (min 0.05, max 0.29)
