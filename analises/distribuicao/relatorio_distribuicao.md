# Distribuicao dos conjuntos (VinDr-Mammo, so imagens com massa)

Gerado por analise_distribuicao.py. Numeros agregados; nenhum ID ou dado de paciente.
KS = Kolmogorov-Smirnov (2 amostras); p < 0,05 sugere distribuicoes diferentes.
Com n pequeno (val: 92 imagens / 97 massas) os testes tem pouco poder: p alto nao prova igualdade.

## Massas por imagem

- train: n=802  media=1.1  mediana=1.0  Q1-Q3=1.0-1.0  min=1.0  max=4.0
- val: n=92  media=1.1  mediana=1.0  Q1-Q3=1.0-1.0  min=1.0  max=2.0
- test: n=219  media=1.1  mediana=1.0  Q1-Q3=1.0-1.0  min=1.0  max=4.0
- KS train x val: D=0.039, p=0.999
- KS train x test: D=0.030, p=0.997
- KS val x test: D=0.014, p=1.000

## Tamanho da massa: raiz da area da caixa (px do DICOM)

- train: n=892  media=266.5  mediana=237.7  Q1-Q3=159.9-344.0  min=41.9  max=947.1
- val: n=97  media=276.9  mediana=262.6  Q1-Q3=173.3-329.2  min=86.8  max=731.1
- test: n=237  media=249.4  mediana=224.7  Q1-Q3=155.9-312.7  min=72.3  max=771.6
- KS train x val: D=0.100, p=0.320
- KS train x test: D=0.098, p=0.050
- KS val x test: D=0.192, p=0.011

## Tamanho da massa: raiz da area da caixa (mm)

- train: n=892  media=22.5  mediana=20.1  Q1-Q3=13.6-29.0  min=3.6  max=80.5
- val: n=97  media=23.4  mediana=22.3  Q1-Q3=14.7-27.7  min=7.4  max=62.1
- test: n=237  media=21.0  mediana=19.0  Q1-Q3=13.1-26.3  min=6.1  max=62.8
- KS train x val: D=0.100, p=0.322
- KS train x test: D=0.091, p=0.081
- KS val x test: D=0.182, p=0.018

## Faixas de tamanho do COCO (as usadas em mAP_s / mAP_m / mAP_l)

| categoria | train (n=892) | val (n=97) | test (n=237) |
|---|---|---|---|
| grande (>=96px) | 859 (96.3%) | 96 (99.0%) | 227 (95.8%) |
| media (32-96px) | 33 (3.7%) | 1 (1.0%) | 10 (4.2%) |

Qui-quadrado (3 conjuntos): chi2=2.14, gl=2, p=0.343 (atencao: ha caselas com esperado < 5; p aproximado)

## BI-RADS da massa (finding_birads)

| categoria | train (n=892) | val (n=97) | test (n=237) |
|---|---|---|---|
| BI-RADS 3 | 404 (45.3%) | 41 (42.3%) | 123 (51.9%) |
| BI-RADS 4 | 362 (40.6%) | 46 (47.4%) | 88 (37.1%) |
| BI-RADS 5 | 126 (14.1%) | 10 (10.3%) | 26 (11.0%) |

Qui-quadrado (3 conjuntos): chi2=6.04, gl=4, p=0.196

## BI-RADS da mama (breast_birads)

| categoria | train (n=802) | val (n=92) | test (n=219) |
|---|---|---|---|
| BI-RADS 3 | 354 (44.1%) | 40 (43.5%) | 111 (50.7%) |
| BI-RADS 4 | 305 (38.0%) | 40 (43.5%) | 80 (36.5%) |
| BI-RADS 5 | 143 (17.8%) | 12 (13.0%) | 28 (12.8%) |

Qui-quadrado (3 conjuntos): chi2=5.91, gl=4, p=0.206

## Densidade mamaria (breast_density)

| categoria | train (n=802) | val (n=92) | test (n=219) |
|---|---|---|---|
| DENSITY A | 0 (0.0%) | 5 (5.4%) | 0 (0.0%) |
| DENSITY B | 95 (11.8%) | 12 (13.0%) | 25 (11.4%) |
| DENSITY C | 670 (83.5%) | 66 (71.7%) | 171 (78.1%) |
| DENSITY D | 37 (4.6%) | 9 (9.8%) | 23 (10.5%) |

Qui-quadrado (3 conjuntos): chi2=69.13, gl=6, p=0.000 (atencao: ha caselas com esperado < 5; p aproximado)

## Lado

| categoria | train (n=802) | val (n=92) | test (n=219) |
|---|---|---|---|
| L | 416 (51.9%) | 58 (63.0%) | 122 (55.7%) |
| R | 386 (48.1%) | 34 (37.0%) | 97 (44.3%) |

Qui-quadrado (3 conjuntos): chi2=4.65, gl=2, p=0.098

## Incidencia

| categoria | train (n=802) | val (n=92) | test (n=219) |
|---|---|---|---|
| CC | 376 (46.9%) | 44 (47.8%) | 104 (47.5%) |
| MLO | 426 (53.1%) | 48 (52.2%) | 115 (52.5%) |

Qui-quadrado (3 conjuntos): chi2=0.05, gl=2, p=0.976

## Fabricante (DICOM Manufacturer)

| categoria | train (n=802) | val (n=92) | test (n=219) |
|---|---|---|---|
| IMS GIOTTO S.p.A. | 10 (1.2%) | 0 (0.0%) | 6 (2.7%) |
| IMS s.r.l. | 117 (14.6%) | 12 (13.0%) | 29 (13.2%) |
| Planmed | 111 (13.8%) | 9 (9.8%) | 38 (17.4%) |
| SIEMENS | 564 (70.3%) | 71 (77.2%) | 146 (66.7%) |

Qui-quadrado (3 conjuntos): chi2=8.32, gl=6, p=0.216 (atencao: ha caselas com esperado < 5; p aproximado)

## Modelo do equipamento (DICOM ManufacturerModelName)

| categoria | train (n=802) | val (n=92) | test (n=219) |
|---|---|---|---|
| GIOTTO CLASS | 37 (4.6%) | 3 (3.3%) | 9 (4.1%) |
| GIOTTO IMAGE 3DL | 90 (11.2%) | 9 (9.8%) | 26 (11.9%) |
| Mammomat Inspiration | 564 (70.3%) | 71 (77.2%) | 146 (66.7%) |
| Planmed Nuance | 111 (13.8%) | 9 (9.8%) | 38 (17.4%) |

Qui-quadrado (3 conjuntos): chi2=4.55, gl=6, p=0.603 (atencao: ha caselas com esperado < 5; p aproximado)

## Tamanho do pixel (mm)

- train: n=802  media=0.1  mediana=0.1  Q1-Q3=0.1-0.1  min=0.1  max=0.1
- val: n=92  media=0.1  mediana=0.1  Q1-Q3=0.1-0.1  min=0.1  max=0.1
- test: n=219  media=0.1  mediana=0.1  Q1-Q3=0.1-0.1  min=0.1  max=0.1
- KS train x val: D=0.028, p=1.000
- KS train x test: D=0.013, p=1.000
- KS val x test: D=0.029, p=1.000

## Altura da imagem (px)

- train: n=802  media=3430.1  mediana=3518.0  Q1-Q3=3518.0-3518.0  min=2812.0  max=3580.0
- val: n=92  media=3457.0  mediana=3518.0  Q1-Q3=3518.0-3518.0  min=2812.0  max=3580.0
- test: n=219  media=3405.4  mediana=3518.0  Q1-Q3=3518.0-3518.0  min=2812.0  max=3580.0
- KS train x val: D=0.041, p=0.998
- KS train x test: D=0.035, p=0.978
- KS val x test: D=0.076, p=0.818

## Largura da imagem (px)

- train: n=802  media=2687.1  mediana=2800.0  Q1-Q3=2800.0-2800.0  min=2012.0  max=2812.0
- val: n=92  media=2720.5  mediana=2800.0  Q1-Q3=2800.0-2800.0  min=2012.0  max=2812.0
- test: n=219  media=2660.2  mediana=2800.0  Q1-Q3=2800.0-2800.0  min=2012.0  max=2812.0
- KS train x val: D=0.062, p=0.890
- KS train x test: D=0.038, p=0.955
- KS val x test: D=0.088, p=0.662
