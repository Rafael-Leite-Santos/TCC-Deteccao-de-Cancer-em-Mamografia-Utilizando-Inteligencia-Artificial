# Varias execucoes por configuracao (ponto 6 do orientador, 16-17/11/2026)

Sementes fixas 0, 1, 2 (`configs/sementes/`, `randomness=dict(seed=N, deterministic=False)`) somadas as
execucoes anteriores sem semente fixa da mesma configuracao. Teste oficial do VinDr (219 imagens, 237 massas).
Media +- desvio padrao amostral; IC 95% pela distribuicao t. Comparacao com a referencia: teste t de Welch.
O `r50rot_s1` falhou na 1a tentativa (log 20261117_084530 para logo apos "Checkpoints will be saved") e foi refeito em 17/11 (20261117_174838).

Execucoes (val mAP@50 na melhor epoca / teste mAP@50 / teste mAP / teste mAP@75):
- ResNet-50 ref.: original 20261028_233314 (70,5 / 58,7 / 27,7 / 24,6); wd1e-4 20261116_013427 (69,6 / 60,6 / 29,7 / 24,5);
  s0 (70,0 / 59,6 / 30,0 / 24,2); s1 (71,9 / 62,5 / 29,6 / 21,9); s2 (70,1 / 59,0 / 28,5 / 23,2)
- ResNet-18: 20261116_152840 (73,2 / 62,6 / 30,0 / 23,7); s0 (71,2 / 59,9 / 27,2 / 18,2); s1 (73,6 / 60,8 / 27,9 / 17,5); s2 (70,2 / 59,5 / 27,7 / 22,3)
- ResNet-50 + rotacao: 20261112_224538 (70,6 / 62,3 / 31,0 / 27,1); s0 (71,1 / 61,1 / 30,7 / 26,7); s1 (72,6 / 60,6 / 30,6 / 26,4); s2 (71,9 / 60,1 / 30,8 / 28,7)

| Configuracao | n | Val mAP@50 | Teste mAP@50 (IC 95%) | Teste mAP (IC 95%) | Teste mAP@75 (IC 95%) |
|---|---|---|---|---|---|
| ResNet-50 (ref.) | 5 | 70,4 +- 0,9 | 60,1 +- 1,5 (58,2-62,0) | 29,1 +- 1,0 (27,9-30,3) | 23,7 +- 1,1 (22,3-25,1) |
| ResNet-18 | 4 | 72,0 +- 1,6 | 60,7 +- 1,4 (58,5-62,9) | 28,2 +- 1,2 (26,2-30,2) | 20,4 +- 3,0 (15,6-25,3) |
| ResNet-50 + rotacao | 4 | 71,5 +- 0,9 | 61,0 +- 0,9 (59,5-62,5) | 30,8 +- 0,2 (30,5-31,0) | 27,2 +- 1,0 (25,6-28,9) |

Diferenca em relacao a ResNet-50 (Welch):
- ResNet-18: val +1,6 (p=0,138); teste mAP@50 +0,6 (p=0,545); mAP -0,9 (p=0,280); mAP@75 -3,3 (p=0,118)
- Rotacao: val +1,1 (p=0,101); teste mAP@50 +0,9 (p=0,296); mAP +1,7 (p=0,017); mAP@75 +3,5 (p=0,002)
Foram feitos 8 testes; com correcao de Bonferroni (0,05/8 = 0,006) so o mAP@75 da rotacao continua significativo.

Grafico: `sementes_teste.png`.

## Pacote de aumento de dados (ponto 3 do orientador, 17-18/11/2026)

`configs/sementes/r50aug_s0..2.py` = rotacao + escala (RandomResize 0,8-1,2) + translacao (ate 10%) +
brilho e contraste (0,8-1,2). Execucoes (val mAP@50 na melhor epoca / teste mAP@50 / mAP / mAP@75; melhor ep. / parou):
- s0 20261117_225608 (77,0 / 65,1 / 33,1 / 26,9; ep. 20 / 28)
- s1 20261118_004055 (73,1 / 63,3 / 31,3 / 28,9; ep. 15 / 19)
- s2 20261118_015206 (77,2 / 65,0 / 33,5 / 29,8; ep. 25 / 33)
(1a tentativa em 17/11 22:45 falhou por disco C: cheio; nao conta.)

| Configuracao | n | Val mAP@50 | Teste mAP@50 (IC 95%) | Teste mAP (IC 95%) | Teste mAP@75 (IC 95%) |
|---|---|---|---|---|---|
| ResNet-50 + pacote de aumento | 3 | 75,8 +- 2,3 | 64,5 +- 1,0 (62,0-67,0) | 32,6 +- 1,2 (29,7-35,5) | 28,5 +- 1,5 (24,8-32,2) |

Diferenca (Welch):
- x ResNet-50 ref.: val +5,3 (p=0,047); teste mAP@50 +4,4 (p=0,003); mAP +3,5 (p=0,014); mAP@75 +4,9 (p=0,012)
- x ResNet-50 + rotacao: val +4,2 (p=0,075); teste mAP@50 +3,4 (p=0,009); mAP +1,9 (p=0,109); mAP@75 +1,3 (p=0,272)
As 3 execucoes do pacote (63,3-65,1%) ficaram acima de todas as 5 da referencia (58,7-62,5%) e das 4 da rotacao (60,1-62,3%) em mAP@50 no teste.
