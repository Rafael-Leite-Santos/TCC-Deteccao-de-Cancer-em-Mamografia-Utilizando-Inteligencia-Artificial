"""Backend da interface web do TCC (detecção de massas em mamografia, VinDr-Mammo).

Importa as funções do VinDR_inference.py, sem alterar o script, que continua
funcionando pela linha de comando como antes.

Rodar (1 worker, sem --reload, para carregar o modelo uma vez só):
    python -m uvicorn main:app --port 8000

Variáveis de ambiente (todas opcionais):
    CONFIG      config do MMDetection   (padrão: CONFIG_PADRAO do VinDR_inference.py)
    CHECKPOINT  arquivo .pth            (padrão: melhor_checkpoint(cfg.work_dir))
    WORK_DIRS   pasta com os work_dirs do treino (ex.: .../Digital_Eye_DICOM/work_dirs),
                para achar o best_*.pth quando o clone não tem os pesos
    VINDR_ROOT  pasta do VinDr-Mammo (com o finding_annotations.csv), para mostrar as
                massas anotadas (padrão: VINDR_ROOT do config)
    ENSEMBLE    1 = Faster R-CNN + FCOS + ATSS fundidos com WBF (VinDR_ensemble.py);
                os checkpoints vêm de WORK_DIRS (CONFIG e CHECKPOINT são ignorados).
                Desde 19/11/2026 o ensemble é o PADRÃO quando nem CONFIG nem CHECKPOINT
                estão definidos; ENSEMBLE=0 força o modelo único (CONFIG_PADRAO).
                Com o ensemble, a interface oferece um seletor: ensemble ou cada modelo
                sozinho (os 3 já estão carregados, sem custo extra de GPU), com as
                métricas de resultados_ensemble/resultado_{val,test}.json.
    PESOS       pesos do WBF no ensemble, ex.: 1,1,1 (padrão: PESOS_PADRAO do ensemble)
    THR_MIN     score mínimo devolvido  (padrão: 0.05; o front filtra o resto)
    MODO_DEMO   1 = não carrega o modelo e devolve caixas falsas
"""

import base64
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import zlib
from contextlib import asynccontextmanager

import numpy as np
import pydicom
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from metricas import THR_PADRAO  # só Python puro: funciona também no modo demo

log = logging.getLogger("uvicorn.error")
PASTA = os.path.dirname(os.path.abspath(__file__))
THR_MIN = float(os.environ.get("THR_MIN", "0.05"))

# O VinDR_inference importa mmdet, torch, cv2 e VinDR_pipeline no topo.
# Sem eles (ou com versões incompatíveis do mmcv/mmdet), o servidor sobe em modo demo.
try:
    import VinDR_inference as vi
    ERRO_IMPORT = None
except (Exception, SystemExit) as e:  # ImportError, SystemExit e erros de versão do mmcv
    vi = None
    ERRO_IMPORT = f"{type(e).__name__}: {e}"

estado = {
    "modo_demo": True,
    "motivo_demo": None,
    "config": None,
    "checkpoint": None,
    "dispositivo": "cpu",
    "modelo": None,
    "detector": None,   # tipo do detector no config (ex.: FasterRCNN, FCOS, ATSS)
    "recorte": False,   # True se o config usa LoadVindrDicom(recortar=True)
    "anotacoes": None,  # image_id -> [[x1, y1, x2, y2], ...] (do finding_annotations.csv)
    "ensemble": None,   # com ENSEMBLE=1: modelos de VinDR_ensemble.carregar_modelos()
    "pesos": None,      # pesos do WBF no ensemble
    "opcoes": [],       # opções do seletor de modelo (ensemble e cada modelo), com métricas
}
OPCAO_ENSEMBLE = "ensemble"
NOME_ENSEMBLE_RESULTADOS = "ENSEMBLE (WBF)"  # nome da linha do ensemble nos resultado_*.json
ve = None  # VinDR_ensemble, importado só com ENSEMBLE=1
trava = threading.Lock()  # uma inferência por vez na GPU

@asynccontextmanager
async def ciclo_de_vida(_app):
    carregar_modelo()
    yield


app = FastAPI(title="TCC — Detecção de Massas em Mamografia", lifespan=ciclo_de_vida)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def entrar_em_modo_demo(motivo):
    estado.update(modo_demo=True, motivo_demo=motivo, modelo=None, detector=None, recorte=False,
                  ensemble=None, pesos=None)
    log.warning("=" * 70)
    log.warning("MODO DEMONSTRAÇÃO: as caixas devolvidas são FALSAS.")
    log.warning("Motivo: %s", motivo)
    log.warning("=" * 70)


def carregar_modelo():
    """Carrega o modelo UMA vez, quando o servidor sobe."""
    if os.environ.get("MODO_DEMO") == "1":
        return entrar_em_modo_demo("variável MODO_DEMO=1")
    if vi is None:
        return entrar_em_modo_demo(f"não foi possível importar VinDR_inference ({ERRO_IMPORT})")
    ens = os.environ.get("ENSEMBLE")
    if ens == "1" or (ens != "0" and not os.environ.get("CONFIG") and not os.environ.get("CHECKPOINT")):
        return carregar_ensemble()
    try:
        config = os.environ.get("CONFIG") or vi.CONFIG_PADRAO
        cfg = vi.Config.fromfile(config)
        work_dir = cfg.work_dir
        if os.environ.get("WORK_DIRS"):  # mesma subpasta, dentro da pasta de work_dirs do treino
            work_dir = os.path.join(os.environ["WORK_DIRS"], os.path.basename(os.path.normpath(work_dir)))
        # melhor_checkpoint() chama SystemExit se não achar o .pth; por isso o except abaixo
        checkpoint = os.environ.get("CHECKPOINT") or vi.melhor_checkpoint(work_dir)
        dispositivo = "cuda:0" if vi.torch.cuda.is_available() else "cpu"
        log.info("Carregando modelo: %s (%s)", checkpoint, dispositivo)
        modelo = vi.init_detector(config, checkpoint, device=dispositivo)
        anotacoes = vi.ler_anotacoes(os.environ.get("VINDR_ROOT") or cfg.get("VINDR_ROOT"))
        recorte = vi.usa_recorte(modelo)
    except (Exception, SystemExit) as e:
        return entrar_em_modo_demo(f"falha ao carregar o modelo ({type(e).__name__}: {e})")
    estado.update(modo_demo=False, motivo_demo=None, config=config, checkpoint=checkpoint,
                  dispositivo=dispositivo, modelo=modelo, anotacoes=anotacoes,
                  detector=cfg.model.type, recorte=recorte,
                  opcoes=[{"id": "unico", "nome": cfg.model.type, "checkpoint": checkpoint,
                           "metricas": None, "metricas_ok": False}])
    log.info("Detector: %s | recorte da mama: %s", cfg.model.type, "sim" if recorte else "não")
    log.info("Modelo pronto.")


def carregar_ensemble():
    """ENSEMBLE=1: os 3 modelos do VinDR_ensemble.py, fundidos com WBF a cada imagem."""
    global ve
    try:
        import VinDR_ensemble as ve
        dispositivo = "cuda:0" if vi.torch.cuda.is_available() else "cpu"
        pesos = ([float(p) for p in os.environ["PESOS"].split(",")] if os.environ.get("PESOS")
                 else list(ve.PESOS_PADRAO))
        if len(pesos) != len(ve.MODELOS):
            raise ValueError(f"PESOS precisa de {len(ve.MODELOS)} valores (um por modelo), veio {pesos}")
        modelos = ve.carregar_modelos(dispositivo)  # usa WORK_DIRS, se definida
        for nome, _, _, m in modelos:
            if vi.usa_recorte(m):
                raise ValueError(f"o config do {nome} usa recorte da mama; o ensemble não trata recorte")
        cfg = vi.Config.fromfile(modelos[0][1])
        anotacoes = vi.ler_anotacoes(os.environ.get("VINDR_ROOT") or cfg.get("VINDR_ROOT"))
    except (Exception, SystemExit) as e:
        return entrar_em_modo_demo(f"falha ao carregar o ensemble ({type(e).__name__}: {e})")
    tipos = " + ".join(m.cfg.model.type for _, _, _, m in modelos)
    nome_ens = "Ensemble WBF (" + " + ".join(n for n, _, _, _ in modelos) + ")"
    opcoes = [{"id": OPCAO_ENSEMBLE, "nome": nome_ens, "checkpoint": [k for _, _, k, _ in modelos],
               "nome_resultados": NOME_ENSEMBLE_RESULTADOS}]
    opcoes += [{"id": n, "nome": f"{n} (sozinho)", "checkpoint": k, "nome_resultados": n}
               for n, _, k, _ in modelos]
    ler_metricas(opcoes, {n: k for n, _, k, _ in modelos}, pesos)
    estado.update(modo_demo=False, motivo_demo=None, dispositivo=dispositivo, modelo=None,
                  ensemble=modelos, pesos=pesos, anotacoes=anotacoes, recorte=False,
                  detector=nome_ens, opcoes=opcoes,
                  config=[c for _, c, _, _ in modelos], checkpoint=[k for _, _, k, _ in modelos])
    log.info("Ensemble pronto: %s | pesos %s", tipos, pesos)


def ler_metricas(opcoes, ckpts, pesos):
    """Métricas de cada opção, lidas de resultados_ensemble/resultado_{val,test}.json
    (gerados pelo VinDR_ensemble.py). Só valem se o cache de detecções que gerou esses
    resultados registra os MESMOS checkpoints carregados agora (metricas_ok)."""
    pasta = os.path.join(PASTA, ve.PASTA_SAIDA)
    por_split, ok = {}, True
    for split in ("val", "test"):
        try:
            with open(os.path.join(pasta, f"resultado_{split}.json"), encoding="utf-8") as f:
                r = json.load(f)
            por_split[split] = {l["modelo"]: l for l in r["resultados"]}
            por_split[split]["_pesos"] = r.get("pesos")
            with open(os.path.join(pasta, f"deteccoes_{split}.json"), encoding="utf-8") as f:
                reg = json.load(f).get("_checkpoints") or {}
            if {n: os.path.normpath(c) for n, c in reg.items()} != {n: os.path.normpath(c) for n, c in ckpts.items()}:
                ok = False
        except (OSError, ValueError, KeyError) as e:
            log.warning("Métricas de %s indisponíveis: %s", split, e)
            ok = False
    pesos_ok = all(por_split.get(s, {}).get("_pesos") in (None, list(pesos))
                   for s in por_split)
    for o in opcoes:
        o["metricas"] = {s: {k: v for k, v in por_split[s].get(o["nome_resultados"], {}).items() if k != "modelo"}
                         for s in por_split} or None
        o["metricas_ok"] = ok and pesos_ok
    if not ok:
        log.warning("As métricas exibidas NÃO correspondem aos checkpoints carregados: rode "
                    "python VinDR_ensemble.py --split val/test --refazer")


def mapear_para_dicom(caminho_dcm, largura_png, altura_png):
    """Relação entre as coordenadas da imagem_png e o DICOM original.

    Convenção:  x_dicom = offset_x + x_png * escala
                y_dicom = offset_y + y_png * escala

    Hoje o carregar_vindr() só apaga o fundo (não recorta nem redimensiona),
    então é 1:1 (confirmado pelo autor do TCC).

    SE O RECORTE DA MAMA ENTRAR NO PIPELINE, mude SÓ esta função: calcule o canto
    do recorte (offset_x, offset_y) para ESTA imagem e devolva "coordenadas": "mapeado".
    O front só mostra as anotações dos radiologistas (que estão em coordenadas do
    DICOM original) quando "coordenadas" for "1:1".
    """
    return {"coordenadas": "1:1", "offset_x": 0, "offset_y": 0, "escala": 1}


def png_base64(img):
    """PNG sem perda da imagem (numpy), em base64."""
    if vi is not None:
        ok, buf = vi.cv2.imencode(".png", img)
        if not ok:
            raise RuntimeError("falha ao codificar o PNG")
        return base64.b64encode(buf.tobytes()).decode()
    from PIL import Image  # só no modo demo, quando o cv2 não está disponível
    saida = io.BytesIO()
    Image.fromarray(img).save(saida, format="PNG")
    return base64.b64encode(saida.getvalue()).decode()


def detectar_demo(caminho_dcm, image_id):
    """Imagem do DICOM com windowing simples + caixas falsas (sempre as mesmas por arquivo)."""
    try:  # pydicom 3.x
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError:  # pydicom 2.x
        from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

    ds = pydicom.dcmread(caminho_dcm)
    arr = ds.pixel_array
    if arr.ndim == 3:  # multiframe: primeiro frame
        arr = arr[0]
    arr = apply_voi_lut(apply_modality_lut(arr, ds), ds).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    img = ((arr - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8)
    if str(ds.get("PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        img = 255 - img

    h, w = img.shape
    rng = np.random.default_rng(zlib.crc32(image_id.encode()))
    lado = min(h, w)
    deteccoes = []
    for score in (0.87, 0.46, 0.18):  # uma abaixo de 0.30 para testar o controle deslizante
        bw = lado * rng.uniform(0.07, 0.14)
        bh = bw * rng.uniform(0.8, 1.2)
        x1 = rng.uniform(0.15, 0.6) * w
        y1 = rng.uniform(0.2, 0.65) * h
        deteccoes.append(([x1, y1, x1 + bw, y1 + bh], score))
    x1, y1, x2, y2 = deteccoes[0][0]
    anotacoes = [[x1 - lado * 0.01, y1 + lado * 0.012, x2 + lado * 0.008, y2 + lado * 0.015]]
    return img, deteccoes, 1, anotacoes


@app.get("/")
def pagina():
    return FileResponse(os.path.join(PASTA, "index.html"))


@app.get("/status")
def status():
    return {
        "modo_demo": estado["modo_demo"],
        "motivo_demo": estado["motivo_demo"],
        "modelo": estado["detector"],
        "recorte": estado["recorte"],
        "config": estado["config"],
        "checkpoint": estado["checkpoint"],
        "dispositivo": estado["dispositivo"],
        "ensemble": estado["ensemble"] is not None,
        "pesos": estado["pesos"],
        "opcoes": [{k: v for k, v in o.items() if k != "nome_resultados"} for o in estado["opcoes"]],
        "opcao_padrao": estado["opcoes"][0]["id"] if estado["opcoes"] else None,
        "anotacoes_carregadas": estado["anotacoes"] is not None,
        "thr_min": THR_MIN,
        "thr_padrao": THR_PADRAO,  # valor inicial do controle deslizante (o mesmo da CLI)
    }


@app.post("/analisar")
def analisar(arquivo: UploadFile = File(...), modelo: str = Form(OPCAO_ENSEMBLE)):  # "def": roda numa thread
    # modelo: id de uma das opções do /status ("ensemble", "Faster R-CNN", "FCOS", "ATSS");
    # ignorado no modo de modelo único e no modo demo
    nome = os.path.basename(arquivo.filename or "imagem.dicom")
    image_id, ext = os.path.splitext(nome)
    fd, caminho = tempfile.mkstemp(suffix=ext if ext.lower() in (".dcm", ".dicom") else ".dicom")
    try:
        with os.fdopen(fd, "wb") as destino:
            shutil.copyfileobj(arquivo.file, destino)
        try:
            pydicom.dcmread(caminho, stop_before_pixels=True)
        except Exception:
            raise HTTPException(status_code=400, detail="O arquivo enviado não é um DICOM válido.")

        opcao = estado["opcoes"][0] if estado["opcoes"] else None
        if estado["ensemble"] is not None:
            opcao = next((o for o in estado["opcoes"] if o["id"] == modelo), None)
            if opcao is None:
                raise HTTPException(status_code=400, detail=f"Modelo desconhecido: {modelo}")
        try:
            with trava:
                inicio = time.perf_counter()
                if estado["modo_demo"]:
                    img, deteccoes, descartadas, anotacoes = detectar_demo(caminho, image_id)
                else:
                    if estado["ensemble"] is not None:
                        so = None if opcao["id"] == OPCAO_ENSEMBLE else opcao["id"]
                        img, deteccoes, descartadas = ve.detectar_ensemble(
                            estado["ensemble"], caminho, THR_MIN, pesos=estado["pesos"], so=so)
                    else:
                        img, deteccoes, descartadas = vi.detectar(estado["modelo"], caminho, THR_MIN,
                                                                  recortar=estado["recorte"])
                    gt = estado["anotacoes"]
                    anotacoes = gt.get(image_id) if gt is not None else None
                tempo = time.perf_counter() - inicio
            altura, largura = img.shape[:2]
            imagem = png_base64(img)
        except Exception as e:
            log.exception("Erro na inferência")
            raise HTTPException(status_code=500, detail=f"Erro na inferência: {e}")

        return {
            "image_id": image_id,
            "largura": largura,
            "altura": altura,
            "imagem_png": imagem,
            "deteccoes": [
                {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "score": round(s, 4), "label": "mass"}
                for (x1, y1, x2, y2), s in deteccoes
            ],
            # Do finding_annotations.csv: coordenadas do DICOM ORIGINAL (iguais às da
            # imagem_png enquanto mapear_para_dicom() for 1:1)
            "anotacoes": None if anotacoes is None else [
                {"x1": a[0], "y1": a[1], "x2": a[2], "y2": a[3]} for a in anotacoes
            ],
            "descartadas": descartadas,
            "tempo_s": round(tempo, 3),
            "dispositivo": estado["dispositivo"],
            "modelo": opcao["nome"] if (opcao and not estado["modo_demo"]) else estado["detector"],
            "opcao": opcao["id"] if opcao else None,
            "modo_demo": estado["modo_demo"],
            **mapear_para_dicom(caminho, largura, altura),
        }
    finally:
        os.remove(caminho)
