# CORREÇÕES E PONTOS DE ATENÇÃO

> **Status em 08/11/2026 (branch `limpeza-e-interface`).** O texto original, de 05/11,
> foi mantido abaixo como histórico; cada seção ganhou uma linha **Status**.
>
> | # | Assunto | Status |
> |---|---|---|
> | 1 | Coordenadas com o recorte da mama | **Resolvido.** O `detectar()` devolve as caixas no DICOM original, e o `main.py` repassa o recorte |
> | 2 | Config padrão, `configs/configs/`, `val_loss_hook.py` | **Resolvido.** `work_dir` relativo: **ainda vale** |
> | 3 | `melhor_checkpoint()` com `SystemExit` | **Ainda vale** (tratado no `main.py`) |
> | 4 | Rótulo "mass" × "massa" | **Ainda vale** (decisão mantida) |
> | 5 | TPR/FPPI calculado de dois jeitos (novo) | **Resolvido** em `metricas.py` |
>
> A frase original "Nenhuma linha do `VinDR_inference.py` foi alterada" **deixou de valer**:
> o script ganhou `metricas.py`, o parâmetro `filtros` no `detectar()` e a opção `--filtros`.

**Nenhuma linha do `VinDR_inference.py` foi alterada.** A interface web (`main.py`)
importa as funções dele, e o uso pelo terminal continua igual.
Este arquivo lista o que foi encontrado durante a integração e o que ficou decidido.

---

## ⚠️ 1. Métricas (TPR/FPPI) e caixas verdes — OK por enquanto (sem recorte)

**Status (08/11): resolvido.** O recorte entrou como experimento
(`configs/VinDR_faster_rcnn_recorte.py`). O `detectar()` recorta só a entrada do modelo e
soma o canto do recorte às caixas, que saem no DICOM original. Por isso o
`mapear_para_dicom()` continua 1:1, as caixas verdes e as métricas da CLI continuam
corretas, e nenhuma conversão do `finding_annotations.csv` é necessária. Desde o commit
`2bcf884`, o `main.py` passa `recortar=vi.usa_recorte(modelo)` ao `detectar()`, e o
`/status` mostra `"recorte"`.

**O que era:** o `ler_anotacoes()` lê as caixas do `finding_annotations.csv`, que estão
em pixels do **DICOM original**. O `main()` compara essas caixas por IoU com as detecções
e desenha as duas sobre o `img` que o `carregar_vindr()` devolve, sem nenhuma conversão
de coordenadas.

**Situação:** foi confirmado pelo autor e conferido no código do `VinDR_pipeline.py`:
o `carregar_vindr()` **só apaga o fundo** (`cv2.bitwise_and(gray, mascara)`), sem recortar
nem redimensionar. A imagem sai do mesmo tamanho do DICOM, com as mesmas coordenadas, e as
métricas e as caixas verdes estão corretas. **Nada foi mudado.**

**AVISO, se o recorte da mama entrar no pipeline:**
- As caixas do `finding_annotations.csv` precisam ser **convertidas para o sistema da
  imagem recortada** (subtrair o canto do recorte) **antes** do cálculo de IoU e do
  `desenhar()`. Sem isso, a sensibilidade e o FPPI ficam errados e as caixas verdes
  aparecem deslocadas.
- As detecções do modelo precisam ser **convertidas de volta para o DICOM** (somar o
  canto do recorte) para o `deteccoes.csv` continuar em coordenadas do DICOM original.
- Na interface web, basta mudar a função `mapear_para_dicom()` no `main.py`. Enquanto
  ela não devolver `"coordenadas": "1:1"`, o front esconde a camada verde automaticamente.

---

## 2. Config padrão × texto de ajuda

**Status (08/11):**
- **Resolvido:** o padrão agora é `configs/VinDR_faster_rcnn.py` (Faster R-CNN de 40 épocas)
  no `VinDR_train.py`, no `VinDR_inference.py` (código e texto de ajuda) e, por consequência,
  no `main.py`. A "Decisão" abaixo (200 épocas) foi substituída.
- **Resolvido:** `configs/configs/` era engano. Os configs estão em `configs/`, e as
  cópias da raiz foram removidas.
- **Resolvido:** o `val_loss_hook.py` está no repositório. O `verificar.py` só o exige
  quando o config o lista em `custom_imports` (hoje, só o de 200 épocas; commit `dc3854f`).
- **Resolvido:** os configs em `configs/` usam só `/` nos caminhos.
- **Ainda vale:** o `work_dir` é relativo à pasta onde o comando roda. Rode da raiz do clone,
  com o checkpoint em `work_dirs/…`, ou use `--checkpoint` / `CHECKPOINT=` (veja o COMO_RODAR.md).

**O que era:** `CONFIG_PADRAO = 'configs/configs/VinDR_faster_rcnn_200ep.py'`, mas o
texto de ajuda no topo do script diz `padrao: configs/VinDR_faster_rcnn.py`.

**Decisão:** o config correto é o `VinDR_faster_rcnn_200ep` (o último treinado). Quem
está **desatualizado é o texto de ajuda**, não o `CONFIG_PADRAO`.
**Nada foi mudado.** Sugestão: atualizar a linha 10 do script para o nome certo.

**A CONFIRMAR:** o caminho tem `configs/` repetido (`configs/configs/...`). Conferir se
essa é mesmo a estrutura de pastas no PC dele. Se o arquivo estiver em
`configs/VinDR_faster_rcnn_200ep.py`, o script e o servidor só funcionam passando
`--config` (terminal) ou `CONFIG=...` (servidor).

**A CONFIRMAR:** no config, `work_dir = './work_dirs/VinDR_faster_rcnn_200ep_sem_parada'`
é um caminho **relativo**. O `melhor_checkpoint(cfg.work_dir)` procura o `best_*.pth` a partir
da pasta onde o comando é executado, e não da pasta do config. O script e o servidor só acham
o checkpoint se forem rodados na mesma pasta usada no treino (ou com `--checkpoint` / `CHECKPOINT=`).

**A CONFIRMAR:** o config tem `custom_imports = ['VinDR_pipeline', 'val_loss_hook']` com
`allow_failed_imports=False`. O `val_loss_hook.py` não veio junto com os arquivos enviados.
Ele precisa estar na mesma pasta, senão o `Config.fromfile()` falha e o servidor sobe em
modo demonstração (o motivo aparece no log).

Detalhe sem efeito prático: o config mistura `ANNOTS_DIR + r'\val.json'` e
`ANNOTS_DIR + '/val.json'`. No Windows os dois funcionam, mas em Linux/macOS só a barra `/`.
Nada foi mudado.

---

## 3. `melhor_checkpoint()` encerra o programa (SystemExit)

**Status (08/11): ainda vale**, sem mudança. Observação: hoje há uma função
`melhor_checkpoint()` em quatro arquivos, com comportamentos diferentes (`SystemExit`,
`sys.exit` ou `None`).

**O que era:** quando não encontra `best_*.pth`, a função chama `raise SystemExit(...)`.
No terminal isso está certo. Num servidor, porém, `except Exception` **não** pega
`SystemExit`, e o servidor cairia ao subir.

**O que mudou:** nada no script. O `main.py` pega `SystemExit` explicitamente e sobe em
modo demonstração, com um aviso no log.

---

## 4. Rótulo "mass" × "massa"

**Status (08/11): ainda vale** (decisão mantida).

**O que era:** o script escreve `mass` nas caixas, e a interface deve mostrar "massa".

**O que mudou:** nada no script. A API devolve `"label": "mass"`, como o modelo e o script,
e só a interface traduz para "massa".

---

## 5. TPR/FPPI calculados de dois jeitos (registrado em 08/11)

**O que era:** o `VinDR_inference.py` contava como acerto qualquer detecção com IoU ≥ 0,5
em alguma massa (duas caixas na mesma massa não geravam falso positivo) e sempre aplicava
os filtros de "fora da mama" e "caixa grande". O `VinDR_ensemble.py`, que gerou a tabela
do README, pareava 1 para 1 e não aplicava filtros.

**Status: resolvido.** Os dois scripts usam `metricas.tpr_fppi()` (pareamento 1 para 1,
cópia da função do ensemble, com testes em `tests/`). Os filtros viraram o parâmetro
`filtros` do `detectar()`: a interface filtra por padrão, e a CLI só com `--filtros`.
**Os números da CLI mudaram.** O limiar padrão é um só, `metricas.THR_PADRAO = 0,3`, que
também define a posição inicial do controle deslizante da interface.

---

## Observação sobre o `main.py`

O import do `VinDR_inference` pega `ImportError` e `SystemExit`, como pedido, e também
`Exception`. Isso cobre erros comuns de versão do mmcv/mmdet, que costumam aparecer como
`AssertionError`. Em qualquer desses casos o servidor sobe em modo demonstração e o
motivo aparece no log e em `GET /status`.

---

## 6. Caminho das anotacoes no treino dentro do clone (registrado em 12/11)

**O que era:** os configs do VinDr em `configs/` usam `data_root=IMAGES_ROOT` e
`ann_file=ANNOTS_DIR + '/train.json'`, ambos relativos. O mmengine
(`BaseDataset._join_prefix`) junta `data_root` + `ann_file` quando o `ann_file` nao e
absoluto, e procura `data/VinDr-Mammo/images\annotations_VinDR/train.json`, que nao existe.
Erro no primeiro treino no clone: `FileNotFoundError: [Errno 2] No such file or directory:
'data/VinDr-Mammo/images\\annotations_VinDR/train.json'`. Nos treinos originais os caminhos
eram absolutos (`C:/Users/rl-sa/Digital_Eye_DICOM/annotations_VinDR`, ver
`work_dirs/VinDR_faster_rcnn_40ep_parada_map50/VinDR_faster_rcnn.py`) e o mmengine nao juntava.

**O que mudou:** so em `configs/VinDR_faster_rcnn_rot.py`: `data_root=''` e
`data_prefix=dict(img=IMAGES_ROOT)` nos 3 dataloaders. O caminho final das imagens e o mesmo
(`IMAGES_ROOT/<file_name>`) e o `ann_file` passa a ser relativo a raiz do clone.

**Status (12/11): resolvido em todos os configs**, a pedido do Rafael: a mesma troca foi
aplicada em `VinDR_faster_rcnn.py`, `VinDR_faster_rcnn_200ep.py` (inclusive o dataloader do
`ValLossHook`), `VinDR_faster_rcnn_recorte.py`, `VinDR_fcos.py`, `VinDR_atss.py`,
`cbis_faster_rcnn.py` e `cbis_cascade_rcnn.py`. Modelo, pipeline, hiperparametros e `work_dir`
nao mudaram. Os configs salvos dentro de `work_dirs/` (copias dos treinos antigos) nao foram
tocados. Continua valendo: rodar sempre da raiz do clone (caminhos relativos).

---

## 7. Ensemble com o Faster R-CNN com aumento de dados (registrado em 18/11)

**O que era:** `VinDR_ensemble.py` usava `configs/VinDR_faster_rcnn.py` (Faster sem aumento de dados).
O cache `resultados_ensemble/deteccoes_<split>.json` nao registrava quais checkpoints o geraram: trocar um
modelo sem `--refazer` misturaria, sem aviso, deteccoes do modelo antigo. Com `WORK_DIRS` definida,
`pasta_do_modelo()` usava so o ultimo nome do `work_dir`, o que quebra para `work_dirs/sementes/r50aug_s2`.

**O que mudou (pedido do Rafael):**
- `MODELOS[0]` = `('Faster R-CNN', 'configs/sementes/r50aug_s2.py')` (pacote de aumento, semente 2,
  ep. 25, escolhido pela validacao: 77,2%). FCOS e ATSS seguem os mesmos (sem aumento) por enquanto.
- O cache passa a guardar `_checkpoints` (nome -> checkpoint). Se nao bater com os checkpoints atuais,
  o script roda os modelos de novo, mesmo sem `--refazer`.
- `pasta_do_modelo()` com `WORK_DIRS` mantem as subpastas abaixo de `work_dirs` (para os antigos,
  igual a antes).
- `reavaliar_iou.py` ignora a chave `_checkpoints` ao ler o cache.
- Copia dos resultados antigos (ensemble sem aumento) em `resultados_ensemble_v1_sem_aumento/`.

**Efeito:** a interface com `ENSEMBLE=1` (`main.py`, sem alteracao) passa a carregar o Faster com aumento.
Os numeros antigos do ensemble (README, `resultados/`) continuam validos para a configuracao antiga.

---

## 8. Ensemble v3: FCOS e ATSS com aumento de dados (registrado em 19/11)

**O que mudou (pedido do Rafael, etapa 2):** `MODELOS` do `VinDR_ensemble.py` passa a usar
`configs/sementes/fcosaug_s0.py` (val mAP@0,50 79,3%, ep. 23) e `configs/sementes/atssaug_s0.py`
(val 76,2%, ep. 23), escolhidos pela validacao entre 3 sementes. Faster segue `r50aug_s2`.
Resultados do ensemble v2 copiados para `resultados_ensemble_v2_faster_aug/`. O cache e refeito
automaticamente (registro de checkpoints, item 7).

---

## 9. Interface: ensemble v3 como padrao e seletor de modelo com metricas (registrado em 19/11)

**Pedido do Rafael.** Arquivos alterados: `main.py` e `index.html`.

**O que mudou:**
- `main.py`: sem `CONFIG`/`CHECKPOINT`, o servidor sobe com o ensemble (`ENSEMBLE=0` forca o modelo
  unico). `/status` ganha `opcoes` (ensemble + cada modelo do ensemble, com checkpoint e metricas de
  `resultados_ensemble/resultado_{val,test}.json`) e `opcao_padrao`. `metricas_ok` so e verdadeiro se o
  cache `deteccoes_{val,test}.json` registra os mesmos checkpoints carregados e os mesmos pesos.
  `POST /analisar` aceita o campo `modelo` (id da opcao; desconhecido -> 400) e devolve `opcao`.
  Continua: modelos carregados uma vez, 1 worker, `trava` na inferencia, endpoint `def`.
- `VinDR_ensemble.detectar_ensemble(..., so=None)`: com `so='FCOS'` (etc.) roda so esse modelo, sem
  fusao, com os mesmos filtros. Nenhum modelo extra e carregado: os 3 ja estao na GPU para o ensemble.
- `index.html`: card "Modelo" com seletor e tabela (val mAP@0,50; teste mAP@0,50, mAP@0,75, TPR, FPPI)
  e nota de leitura; trocar o modelo refaz a analise da imagem aberta. Arredondamento com +1e-9
  (0,6635 -> 66,4, como no terminal). Fim de linha CRLF preservado.
- Testado: sintaxe Python/JS; `ler_metricas()` com os arquivos reais (metricas_ok = true); layout no
  Chromium com um /status simulado (sem erros no console). NAO TESTADO: servidor com os modelos reais.
