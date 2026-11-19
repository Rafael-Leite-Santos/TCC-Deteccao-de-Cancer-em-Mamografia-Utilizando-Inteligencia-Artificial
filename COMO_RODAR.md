# Como rodar a interface

A interface carrega o **Faster R-CNN de 40 épocas** (`configs/VinDR_faster_rcnn.py`) e o
`best_*.pth` mais recente de `work_dirs/VinDR_faster_rcnn_40ep_parada_map50/`. O ensemble
(`VinDR_ensemble.py`) é um resultado de avaliação e **não está disponível na interface**.

Sem MMDetection, OpenCV ou checkpoint, ela sobe em **modo demo**: as caixas são **falsas**.

> **Use o repositório pelo git** (`git clone` / `git pull`), não copiando arquivos soltos.
> A interface precisa de `main.py`, `index.html`, `verificar.py`, `VinDR_inference.py`,
> `VinDR_pipeline.py` **e `metricas.py`** (novo). Sem o `metricas.py`, o modelo não carrega
> e a interface cai em modo demo.

---

## A) Mac ou Linux sem GPU — modo demo

Serve para ver a interface funcionando, sem o modelo. Não precisa de mmdet.

```shell
git clone https://github.com/Rafael-Leite-Santos/TCC-Deteccao-de-Cancer-em-Mamografia-Utilizando-Inteligencia-Artificial.git
cd TCC-Deteccao-de-Cancer-em-Mamografia-Utilizando-Inteligencia-Artificial
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-web.txt
python gerar_test_dcm.py            # cria um DICOM sintético (test.dcm), se ainda não houver
./iniciar.command
```

- O `iniciar.command` usa o `python3` do terminal (o do venv ativado). **Se houver conda
  no computador, ele usa o conda** e pede o nome de um ambiente: nesse caso, rode direto
  `python -m uvicorn main:app --port 8000` com o venv ativado.
- Ele mostra a verificação e para em `ATENÇÃO: vai abrir em MODO DEMONSTRAÇÃO`: aperte
  **Enter**. O navegador abre em http://localhost:8000.
- Arraste o `test.dcm` para a página.

**Como saber que está em modo demo:**

- o `verificar.py` termina com `⚠️  Vai rodar em MODO DEMO porque: ...`;
- a página mostra uma **faixa amarela** "MODO DEMONSTRAÇÃO: as caixas na tela são FALSAS";
- `curl http://localhost:8000/status` devolve `"modo_demo": true` e o motivo.

---

## B) PC do Rafael (Windows, GPU) — modo real

### 1. Código e ambiente

```shell
git clone https://github.com/Rafael-Leite-Santos/TCC-Deteccao-de-Cancer-em-Mamografia-Utilizando-Inteligencia-Artificial.git
cd TCC-Deteccao-de-Cancer-em-Mamografia-Utilizando-Inteligencia-Artificial
```

(Se já tiver o clone: `git pull`.) Ambiente do treino, como no README:

```shell
python -m venv venv
venv\Scripts\activate
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install -r requirements.txt
pip install -r requirements-web.txt
```

> O `requirements-web.txt` só pede versões mínimas compatíveis com o `requirements.txt`
> (`pydicom>=2.3.1`; `numpy` e `pillow` sem versão): instalá-lo não atualiza o pydicom nem
> o numpy do ambiente do treino.

### 2. Checkpoint e dataset (ficam fora do git)

```
tcc-rafael\
├── work_dirs\
│   └── VinDR_faster_rcnn_40ep_parada_map50\
│       └── best_coco_bbox_mAP_50_epoch_18.pth     ← copie da pasta do treino
└── data\
    └── VinDr-Mammo\
        ├── finding_annotations.csv                ← só para as caixas verdes
        └── images\<study_id>\<image_id>.dicom
```

- O nome do `.pth` acima é o esperado para a melhor época (18, segundo
  `resultados/logs/README.md`). Confira o nome real na pasta do treino.
- O dataset só é necessário para mostrar as anotações dos radiologistas (camada verde) e
  para o teste de aceitação. Sem ele, a interface funciona, só que sem a camada verde.

### 3. Verificar e ligar

```shell
python verificar.py
```

O final deve ser `✅ Pronto para rodar com o MODELO REAL`. Depois:

- **com conda:** dois cliques no `iniciar.bat` (ele ativa o ambiente conda e pergunta o
  nome na primeira vez);
- **com venv** (como acima): no mesmo terminal, com o venv ativado,
  `python -m uvicorn main:app --port 8000`, e abra http://localhost:8000. O `iniciar.bat`
  não ativa venv: se houver conda instalado, ele tenta ativar um ambiente conda.

### 4. Conferir qual checkpoint foi carregado

A pasta `work_dirs\VinDR_faster_rcnn_40ep_parada_map50\` pode ter **mais de uma execução**:
uma interrompida por `WinError 1455` (logo na 1ª época) e a válida (26 épocas, melhor na
18ª; veja `resultados/logs/README.md`). A interface pega o `best_*.pth` **mais recente**
(pela data do arquivo), que pode não ser o certo se os arquivos foram copiados.

Abra http://localhost:8000/status e confira os campos `checkpoint`, `config`, `modelo`
(deve ser `FasterRCNN`) e `recorte` (deve ser `false`). Para fixar o checkpoint:

```shell
set CHECKPOINT=work_dirs\VinDR_faster_rcnn_40ep_parada_map50\best_coco_bbox_mAP_50_epoch_18.pth
python -m uvicorn main:app --port 8000
```

Na CLI, o equivalente é `--checkpoint caminho.pth`.

### 5. Ensemble (Faster R-CNN + FCOS + ATSS) na interface

Com `ENSEMBLE=1`, o servidor carrega os três modelos e, a cada imagem, funde as caixas com
Weighted Boxes Fusion, com as mesmas funções e os mesmos parâmetros do `VinDR_ensemble.py`
(pesos `1,1,1`, IoU 0,55, skip 0,05). Depois aplica os mesmos filtros da interface com um
modelo só (centro fora da mama, caixa maior que 30% da imagem). Cada imagem leva cerca de
três vezes mais tempo.

Os três `best_*.pth` vêm da pasta de work_dirs do treino, indicada por `WORK_DIRS` (o clone
não tem os pesos). `VINDR_ROOT` aponta para o dataset, para a interface mostrar as massas
anotadas (camada verde):

```powershell
$env:WORK_DIRS  = "$HOME\Digital_Eye_DICOM\work_dirs"
$env:VINDR_ROOT = "D:\Dataset_VinDR-Mammo\vindr-mammo-...-1.0.0"
$env:ENSEMBLE   = "1"
python verificar.py
python -m uvicorn main:app --port 8000
```

Em http://localhost:8000/status, `modelo` deve ser `Ensemble WBF (FasterRCNN + FCOS + ATSS)`,
`ensemble` deve ser `true` e `checkpoint` lista os três arquivos. Para voltar ao Faster R-CNN
sozinho, remova a variável (`Remove-Item Env:ENSEMBLE`, ou `set ENSEMBLE=` no `cmd`) e reinicie
o servidor. `WORK_DIRS` também serve para o modelo único, no lugar do `CHECKPOINT`.

---

## C) Mac em modo real — **não testado**

Não há garantia de que funcione. O que falta:

- **MMCV 2.1.0 não tem pacote pronto para macOS**: precisa ser compilado a partir do código-fonte;
- PyTorch 2.1.0 para CPU (o `requirements.txt` foi testado com CUDA 12.1 no Windows);
- o checkpoint (`best_*.pth`), que não está no repositório e tem de vir do PC do Rafael;
- **sem GPU**: a inferência roda na CPU e cada imagem demora bem mais.

---

## Teste de aceitação: CLI × interface

Objetivo: as caixas da interface devem ser **as mesmas** da linha de comando. Rode no PC
do Rafael (cenário B), depois do `python VinDR_prepare.py`, que gera
`annotations_VinDR\test.json`.

1. Copie 3 imagens do teste oficial para uma pasta (os `.dicom` são ignorados pelo git):

   ```shell
   python -c "import json,os,shutil; os.makedirs('aceitacao',exist_ok=True); [shutil.copy(os.path.join('data/VinDr-Mammo/images',i['file_name']),'aceitacao') for i in json.load(open('annotations_VinDR/test.json'))['images'][:3]]"
   ```

2. CLI, **com filtros** (a interface também filtra) e limiar 0,3:

   ```shell
   python VinDR_inference.py aceitacao --filtros --thr 0.3 --saida aceitacao_cli
   ```

   Gera `aceitacao_cli\deteccoes.csv` e um PNG por imagem.

3. Interface: abra cada uma das 3 imagens, deixe o **Score mínimo em 0,30** (é o valor
   inicial) e clique em **CSV das detecções**.

4. Compare: para cada imagem, as linhas (score com 4 casas e coordenadas inteiras) do CSV
   da interface devem ser iguais às do `deteccoes.csv` da CLI.

Sem `--filtros`, a CLI pode mostrar caixas a mais (fora da mama ou grandes demais), e o
teste não bate.

---

## Problemas comuns (o que o `verificar.py` mostra)

| Aparece | O que fazer |
|---|---|
| `❌ mmengine` / `mmdet` / `opencv (cv2): não instalado` | O Python não é o do ambiente do treino. Ative o ambiente (venv ou conda). No `iniciar.bat`, apague o `ambiente_local.txt` e abra de novo, para ele perguntar o nome. |
| `❌ fastapi: não instalado` (ou uvicorn, python-multipart, pydicom, numpy) | `pip install -r requirements-web.txt` (precisa de internet). |
| `❌ metricas.py não encontrado` (ou outro `.py` do projeto) | Faltam arquivos do repositório: rode `git pull` na pasta do clone. |
| `❌ Config não encontrado: configs/VinDR_faster_rcnn.py` | Rode os comandos a partir da raiz do clone (a pasta que tem `configs\`). |
| `❌ val_loss_hook.py não encontrado` | Só aparece com o config de 200 épocas (`CONFIG=configs/VinDR_faster_rcnn_200ep.py`). O arquivo está no repositório: `git pull`. |
| `❌ Nenhum best_*.pth em ...work_dirs\VinDR_faster_rcnn_40ep_parada_map50` | Confira se você está na raiz do clone e se o checkpoint está nessa pasta (B.2), ou passe o caminho com `CHECKPOINT=` (B.4; no Mac/Linux, `export CHECKPOINT=...`). |
| `⚠️  Sem CUDA` | Não é erro. Roda na CPU, só que mais devagar. |
| `⚠️  Vai rodar em MODO DEMO porque: a variável MODO_DEMO=1 está definida` | Remova a variável (`set MODO_DEMO=` no Windows, `unset MODO_DEMO` no Mac/Linux). |

Variáveis de ambiente do servidor (todas opcionais): `CONFIG` (padrão
`configs/VinDR_faster_rcnn.py`), `CHECKPOINT`, `WORK_DIRS` (pasta de work_dirs do treino),
`VINDR_ROOT` (pasta do dataset, para a camada verde), `ENSEMBLE=1` (Faster R-CNN + FCOS + ATSS
com WBF), `PESOS` (pesos do WBF, ex.: `1,1,1`), `THR_MIN` (padrão 0,05: o piso do que o
backend devolve; o controle deslizante começa em 0,30) e `MODO_DEMO=1`.
