# Desempenho por subgrupo: `VinDR_faster_rcnn_40ep_parada_map50`

TPR e FPPI com score >= 0.3 e IoU >= 0,5 (metricas.py); AP VOC com IoU 0,5 (todas as deteccoes do cache, score >= 0,05).
Gerado por analise_subgrupos.py. Numeros agregados; nenhum ID ou dado de paciente.
Grupos com poucas massas: numeros apenas indicativos. p = qui-quadrado (achada x perdida entre grupos).


# Conjunto: val

## Densidade mamaria

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| DENSITY A | 5 | 5 | 100.0% | 1.00 | 94.3% |
| DENSITY B | 12 | 14 | 85.7% | 0.67 | 85.2% |
| DENSITY C | 66 | 69 | 72.5% | 1.11 | 66.8% |
| DENSITY D | 9 | 9 | 66.7% | 0.78 | 59.8% |

Qui-quadrado (achada x perdida): chi2 = 3.11, gl = 3, p = 0.375 (caselas com esperado < 5: p aproximado)

## Fabricante

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| IMS s.r.l. | 12 | 13 | 84.6% | 1.42 | 75.5% |
| Planmed | 9 | 11 | 72.7% | 1.22 | 52.7% |
| SIEMENS | 71 | 73 | 74.0% | 0.92 | 72.3% |

Qui-quadrado (achada x perdida): chi2 = 0.71, gl = 2, p = 0.700 (caselas com esperado < 5: p aproximado)

## Modelo do equipamento

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| GIOTTO CLASS | 3 | 4 | 50.0% | 1.67 | 42.7% |
| GIOTTO IMAGE 3DL | 9 | 9 | 100.0% | 1.33 | 90.4% |
| Mammomat Inspiration | 71 | 73 | 74.0% | 0.92 | 72.3% |
| Planmed Nuance | 9 | 11 | 72.7% | 1.22 | 52.7% |

Qui-quadrado (achada x perdida): chi2 = 4.43, gl = 3, p = 0.218 (caselas com esperado < 5: p aproximado)

## Incidencia

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| CC | 44 | 46 | 84.8% | 0.82 | 77.5% |
| MLO | 48 | 51 | 66.7% | 1.19 | 65.3% |

Fisher (achada x perdida): p = 0.058

## BI-RADS da massa

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| BI-RADS 3 | 40 | 41 | 75.6% | (n/a) | (n/a) |
| BI-RADS 4 | 44 | 46 | 73.9% | (n/a) | (n/a) |
| BI-RADS 5 | 10 | 10 | 80.0% | (n/a) | (n/a) |

Qui-quadrado (achada x perdida): chi2 = 0.17, gl = 2, p = 0.919 (caselas com esperado < 5: p aproximado)

## Tamanho da massa (lado)

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| 150-250 px | 27 | 27 | 70.4% | (n/a) | (n/a) |
| 250-350 px | 30 | 31 | 87.1% | (n/a) | (n/a) |
| <150 px | 15 | 16 | 37.5% | (n/a) | (n/a) |
| >=350 px | 23 | 23 | 91.3% | (n/a) | (n/a) |

Qui-quadrado (achada x perdida): chi2 = 18.11, gl = 3, p = 0.000 (caselas com esperado < 5: p aproximado)


# Conjunto: test

## Densidade mamaria

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| DENSITY B | 25 | 27 | 85.2% | 0.76 | 78.0% |
| DENSITY C | 171 | 187 | 68.4% | 1.57 | 55.7% |
| DENSITY D | 23 | 23 | 65.2% | 1.22 | 59.7% |

Qui-quadrado (achada x perdida): chi2 = 3.43, gl = 2, p = 0.180

## Fabricante

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| IMS GIOTTO S.p.A. | 6 | 11 | 72.7% | 3.00 | 49.9% |
| IMS s.r.l. | 29 | 35 | 71.4% | 1.31 | 64.7% |
| Planmed | 38 | 40 | 75.0% | 2.08 | 64.7% |
| SIEMENS | 146 | 151 | 68.2% | 1.24 | 57.9% |

Qui-quadrado (achada x perdida): chi2 = 0.78, gl = 3, p = 0.854 (caselas com esperado < 5: p aproximado)

## Modelo do equipamento

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| GIOTTO CLASS | 9 | 14 | 71.4% | 2.22 | 54.2% |
| GIOTTO IMAGE 3DL | 26 | 32 | 71.9% | 1.38 | 64.6% |
| Mammomat Inspiration | 146 | 151 | 68.2% | 1.24 | 57.9% |
| Planmed Nuance | 38 | 40 | 75.0% | 2.08 | 64.7% |

Qui-quadrado (achada x perdida): chi2 = 0.77, gl = 3, p = 0.856 (caselas com esperado < 5: p aproximado)

## Incidencia

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| CC | 104 | 114 | 64.0% | 1.30 | 55.2% |
| MLO | 115 | 123 | 75.6% | 1.57 | 62.6% |

Fisher (achada x perdida): p = 0.065

## BI-RADS da massa

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| BI-RADS 3 | 121 | 123 | 71.5% | (n/a) | (n/a) |
| BI-RADS 4 | 80 | 88 | 63.6% | (n/a) | (n/a) |
| BI-RADS 5 | 25 | 26 | 84.6% | (n/a) | (n/a) |

Qui-quadrado (achada x perdida): chi2 = 4.48, gl = 2, p = 0.106

## Tamanho da massa (lado)

| grupo | imagens | massas | TPR | FPPI | AP@0,5 (VOC) |
|---|---|---|---|---|---|
| 150-250 px | 84 | 89 | 71.9% | (n/a) | (n/a) |
| 250-350 px | 49 | 49 | 77.6% | (n/a) | (n/a) |
| <150 px | 51 | 54 | 44.4% | (n/a) | (n/a) |
| >=350 px | 45 | 45 | 88.9% | (n/a) | (n/a) |

Qui-quadrado (achada x perdida): chi2 = 25.94, gl = 3, p = 0.000

