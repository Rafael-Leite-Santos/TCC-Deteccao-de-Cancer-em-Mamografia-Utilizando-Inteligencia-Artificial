"""
verificar.py - confere se está tudo pronto para ligar a interface com o modelo real.

Uso:  python verificar.py
(O iniciar.bat roda isto sozinho antes de ligar o servidor.)

Código de saída: 0 = modelo real | 1 = modo demo | 2 = o servidor não vai iniciar
"""
import ast
import glob
import importlib
import os
import sys

try:  # emojis no console do Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

OK, ERRO, AVISO = "✅", "❌", "⚠️ "
motivos_demo = []        # o que impede o modelo real
motivos_nao_inicia = []  # o que impede até o modo demo


def linha(simbolo, texto, dica=None):
    print(f"  {simbolo} {texto}")
    if dica:
        for parte in dica.split("\n"):
            print(f"       → {parte}")


def importa(modulo):
    """Devolve (ok, versão ou mensagem de erro)."""
    try:
        m = importlib.import_module(modulo)
        return True, getattr(m, "__version__", "")
    except ImportError as e:
        return False, f"não instalado ({e})"
    except Exception as e:  # instalado, mas quebrou ao carregar (ex.: versão do mmcv)
        return False, f"instalado, mas deu erro ao carregar: {type(e).__name__}: {e}"


def valor_no_arquivo(caminho, nome):
    """Lê 'nome = "texto"' de um .py sem executá-lo."""
    try:
        with open(caminho, encoding="utf-8") as f:
            arvore = ast.parse(f.read())
    except (OSError, SyntaxError):
        return None
    for no in arvore.body:
        if isinstance(no, ast.Assign) and any(getattr(t, "id", None) == nome for t in no.targets):
            if isinstance(no.value, ast.Constant) and isinstance(no.value.value, str):
                return no.value.value
    return None


def imports_do_config(caminho):
    """Lista de 'custom_imports = dict(imports=[...])' de um config, sem executá-lo."""
    try:
        with open(caminho, encoding="utf-8") as f:
            arvore = ast.parse(f.read())
    except (OSError, SyntaxError):
        return []
    for no in arvore.body:
        if (isinstance(no, ast.Assign) and any(getattr(t, "id", None) == "custom_imports" for t in no.targets)
                and isinstance(no.value, ast.Call)):
            for kw in no.value.keywords:
                if kw.arg == "imports":
                    try:
                        return list(ast.literal_eval(kw.value))
                    except ValueError:
                        return []
    return []


print()
print("=" * 64)
print("  VERIFICAÇÃO - Detecção de Massas em Mamografia")
print("=" * 64)
print(f"  Pasta atual: {os.getcwd()}")

# ---------------------------------------------------------------- Python
print("\n[1] Python")
linha(OK, f"Python {sys.version.split()[0]} em {sys.executable}")

# ---------------------------------------------------------------- Bibliotecas do modelo
print("\n[2] Bibliotecas do modelo (as mesmas do treino)")
dica_ambiente = ("Este Python não parece ser o do ambiente onde você treinou o modelo.\n"
                 "Apague o arquivo ambiente_local.txt (se existir) e abra o iniciar.bat de novo;\n"
                 "quando ele perguntar, digite o nome do ambiente conda do treino.")
faltou_modelo = False
for modulo, nome in [("torch", "torch"), ("mmengine", "mmengine"), ("mmdet", "mmdet"), ("cv2", "opencv (cv2)")]:
    ok, info = importa(modulo)
    if ok:
        linha(OK, f"{nome} {info}")
    else:
        linha(ERRO, f"{nome}: {info}", None if faltou_modelo else dica_ambiente)
        faltou_modelo = True
        motivos_demo.append(f"{nome} {info.split(' (')[0]}")

# ---------------------------------------------------------------- Bibliotecas da interface
print("\n[3] Bibliotecas da interface web")
dica_pip = "Rode:  python -m pip install -r requirements-web.txt   (precisa de internet)"
for modulos, nome, essencial in [
    (["fastapi"], "fastapi", True),
    (["uvicorn"], "uvicorn", True),
    (["python_multipart", "multipart"], "python-multipart", True),
    (["pydicom"], "pydicom", True),
    (["numpy"], "numpy", True),
    (["PIL"], "pillow", False),
]:
    resultados = [importa(m) for m in modulos]
    ok = next((r for r in resultados if r[0]), None)
    if ok:
        linha(OK, f"{nome} {ok[1]}")
    elif essencial:
        linha(ERRO, f"{nome}: não instalado", dica_pip)
        motivos_nao_inicia.append(f"falta a biblioteca {nome}")
    else:
        linha(AVISO, f"{nome}: não instalado (só faz falta no modo demo)", dica_pip)

# ---------------------------------------------------------------- GPU
print("\n[4] Placa de vídeo (CUDA)")
try:
    import torch
    if torch.cuda.is_available():
        linha(OK, f"CUDA disponível: {torch.cuda.get_device_name(0)}")
    else:
        linha(AVISO, "Sem CUDA: o modelo vai rodar na CPU (funciona, mas cada imagem demora mais).")
except Exception:
    linha(AVISO, "Não deu para verificar (torch não carregou).")

# ---------------------------------------------------------------- Arquivos do projeto
print("\n[5] Arquivos do projeto nesta pasta")
dica_pasta = ("Copie os arquivos do repositório para a pasta onde você treinou o modelo\n"
              "(a que tem VinDR_pipeline.py e work_dirs) e abra o iniciar.bat de lá.")
for arquivo, para_que in [("main.py", "servidor"), ("index.html", "interface"),
                          ("VinDR_inference.py", "modelo"), ("VinDR_pipeline.py", "modelo"),
                          ("metricas.py", "modelo")]:
    if os.path.isfile(arquivo):
        linha(OK, arquivo)
    elif para_que == "modelo":
        linha(ERRO, f"{arquivo} não encontrado", dica_pasta)
        motivos_demo.append(f"falta o {arquivo}")
    else:
        linha(ERRO, f"{arquivo} não encontrado", dica_pasta)
        motivos_nao_inicia.append(f"falta o {arquivo}")

# ---------------------------------------------------------------- Config
print("\n[6] Config do modelo")
config = os.environ.get("CONFIG") or valor_no_arquivo("VinDR_inference.py", "CONFIG_PADRAO")
if not config:
    linha(ERRO, "Não consegui ler o CONFIG_PADRAO do VinDR_inference.py.")
    motivos_demo.append("config não encontrado")
elif os.path.isfile(config):
    linha(OK, f"{config}")
    # O val_loss_hook.py só é necessário se o config o importar (hoje, só o de 200 épocas)
    if "val_loss_hook" in imports_do_config(config):
        if os.path.isfile("val_loss_hook.py"):
            linha(OK, "val_loss_hook.py (importado pelo config)")
        else:
            linha(ERRO, "val_loss_hook.py não encontrado",
                  "O config importa este arquivo (custom_imports). Sem ele o modelo não carrega.\n"
                  "Copie o val_loss_hook.py da pasta onde você treinou para esta pasta.")
            motivos_demo.append("falta o val_loss_hook.py")
else:
    nome = os.path.basename(config)
    achados = [p for p in glob.glob(os.path.join("**", nome), recursive=True) if "work_dirs" not in p]
    dica = f"O VinDR_inference.py procura o config em: {config}"
    if achados:
        dica += (f"\nEncontrei um arquivo com esse nome em: {achados[0]}"
                 f"\nCopie esse arquivo para {os.path.dirname(config)}{os.sep} (crie as pastas se precisar).")
    linha(ERRO, f"Config não encontrado: {config}", dica)
    motivos_demo.append(f"config não encontrado ({config})")
    config = None

# ---------------------------------------------------------------- Checkpoint
def procurar_best(config, rotulo=""):
    """Procura o best_*.pth do config (no work_dir, ou na subpasta de mesmo nome em WORK_DIRS)."""
    work_dir = valor_no_arquivo(config, "work_dir")
    if not work_dir:
        linha(AVISO, f"{rotulo}Não consegui ler o work_dir de {config}; o servidor vai tentar mesmo assim.")
        return
    if os.environ.get("WORK_DIRS"):
        # mantém as subpastas abaixo de work_dirs (ex.: sementes/r50aug_s2), como o VinDR_ensemble.py
        partes = os.path.normpath(work_dir).replace("\\", "/").split("/")
        rel = partes[partes.index("work_dirs") + 1:] if "work_dirs" in partes else partes[-1:]
        pasta = os.path.join(os.environ["WORK_DIRS"], *rel)
        origem = "a variável WORK_DIRS"
    else:
        pasta = os.path.abspath(work_dir)  # relativo à pasta ATUAL, como no script
        origem = None
    candidatos = glob.glob(os.path.join(pasta, "best_*.pth"))
    if candidatos:
        melhor = max(candidatos, key=os.path.getmtime)
        linha(OK, f"{rotulo}{melhor if origem else os.path.relpath(melhor)}"
                  + (f"  (o mais recente de {len(candidatos)})" if len(candidatos) > 1 else "")
                  + (f"  (via {origem})" if origem else ""))
    else:
        linha(ERRO, f"{rotulo}Nenhum best_*.pth em {pasta}",
              f"O config diz work_dir = '{work_dir}'"
              + (", procurado dentro de WORK_DIRS.\n" if origem else
                 ", e esse caminho é contado a partir desta pasta.\n")
              + "No clone, os pesos não vêm junto: aponte para a pasta do treino (veja o COMO_RODAR.md):\n"
              "  Windows:     set WORK_DIRS=caminho\\do\\treino\\work_dirs\n"
              "               ou set CHECKPOINT=caminho\\do\\best_xxx.pth  (só um modelo)\n"
              "  Mac/Linux:   export WORK_DIRS=... / export CHECKPOINT=...")
        motivos_demo.append(f"checkpoint best_*.pth não encontrado ({os.path.basename(config)})")


print("\n[7] Checkpoint do modelo treinado (best_*.pth)")
def configs_do_ensemble():
    """Configs da lista MODELOS do VinDR_ensemble.py, lidos do texto (sem importar mmdet)."""
    import re
    try:
        with open("VinDR_ensemble.py", encoding="utf-8") as f:
            txt = f.read()
        bloco = txt[txt.index("MODELOS = ["):txt.index("]", txt.index("MODELOS = ["))]
        return re.findall(r"'(configs/[^']+\.py)'", bloco)
    except (OSError, ValueError):
        return []


# Mesma regra do main.py (19/11/2026): sem CONFIG/CHECKPOINT, o padrão é o ensemble; ENSEMBLE=0 força o modelo único
_ens = os.environ.get("ENSEMBLE")
if _ens == "1" or (_ens != "0" and not os.environ.get("CONFIG") and not os.environ.get("CHECKPOINT")):
    print("  (ensemble: os modelos da lista MODELOS do VinDR_ensemble.py; CONFIG e CHECKPOINT são ignorados)")
    if not importa("ensemble_boxes")[0]:
        linha(ERRO, "ensemble_boxes não instalado", "É a biblioteca do WBF, a mesma do VinDR_ensemble.py.")
        motivos_demo.append("falta o ensemble_boxes")
    lista = configs_do_ensemble()
    if not lista:
        linha(ERRO, "Não consegui ler a lista MODELOS do VinDR_ensemble.py")
        motivos_demo.append("lista de modelos do ensemble ilegível")
    for cfg_ens in lista:
        if os.path.isfile(cfg_ens):
            procurar_best(cfg_ens, f"{os.path.basename(cfg_ens)}: ")
        else:
            linha(ERRO, f"Config não encontrado: {cfg_ens}")
            motivos_demo.append(f"config não encontrado ({cfg_ens})")
elif os.environ.get("CHECKPOINT"):
    ck = os.environ["CHECKPOINT"]
    if os.path.isfile(ck):
        linha(OK, f"{ck} (definido na variável CHECKPOINT)")
    else:
        linha(ERRO, f"CHECKPOINT aponta para um arquivo que não existe: {ck}")
        motivos_demo.append("checkpoint não encontrado")
elif not config:
    linha(ERRO, "Não dá para procurar sem o config (veja o item 6).")
    motivos_demo.append("checkpoint não verificado")
else:
    procurar_best(config)

if os.environ.get("MODO_DEMO") == "1":
    motivos_demo.insert(0, "a variável MODO_DEMO=1 está definida")

# ---------------------------------------------------------------- Resultado
print("\n" + "=" * 64)
if motivos_nao_inicia:
    print(f"  {ERRO} O SERVIDOR NÃO VAI INICIAR porque: " + "; ".join(motivos_nao_inicia) + ".")
    codigo = 2
elif motivos_demo:
    print(f"  {AVISO} Vai rodar em MODO DEMO porque: " + "; ".join(motivos_demo) + ".")
    print("     As caixas na tela serão FALSAS. Corrija os itens com ❌ acima.")
    codigo = 1
else:
    print(f"  {OK} Pronto para rodar com o MODELO REAL")
    codigo = 0
print("=" * 64 + "\n")
sys.exit(codigo)
