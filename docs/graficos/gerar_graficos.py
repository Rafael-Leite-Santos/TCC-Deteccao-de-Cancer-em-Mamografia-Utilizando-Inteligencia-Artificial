"""Gera os gráficos do ARQUITETURA.md a partir dos logs versionados (scalars.json) e dos
resultado_*.json. Nenhum número é digitado à mão, exceto a tabela de CBIS (D–I), que vem
de resultados/logs/README.md e é conferida contra os scalars de H e I."""
import json, os, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Uso, na raiz do repositorio: python docs/graficos/gerar_graficos.py
RAIZ = sys.argv[1] if len(sys.argv) > 1 else "."
SAIDA = sys.argv[2] if len(sys.argv) > 2 else os.path.join("docs", "graficos")
os.makedirs(SAIDA, exist_ok=True)
LOGS = os.path.join(RAIZ, "resultados", "logs")

AZUL, LARANJA, AQUA, AMARELO, CINZA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8c8a85"
TXT, TXT2, GRADE, FUNDO = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRADE,
    "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRADE, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
    "legend.frameon": False,
})


def ler(rel):
    linhas = [json.loads(l) for l in open(os.path.join(LOGS, rel, "scalars.json"))]
    val = {r["step"]: r for r in linhas if "coco/bbox_mAP_50" in r}
    perda = defaultdict(list)
    for r in linhas:
        if "coco/bbox_mAP_50" not in r and "loss" in r and "epoch" in r:
            perda[r["epoch"]].append(r["loss"])
    perda = {e: sum(v) / len(v) for e, v in perda.items()}
    return val, perda


def pct(v):
    return f"{v * 100:.1f}".replace(".", ",") + "%"


def fppi_txt(v):
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(v)).quantize(Decimal('0.01'), ROUND_HALF_UP)).replace('.', ',')


def salvar(fig, nome):
    fig.savefig(os.path.join(SAIDA, nome), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("ok", nome)


# ---------- G1: mAP@0,50 de validação por época (VinDr) ----------
runs = [
    ("Faster R-CNN", "VinDR_faster_rcnn_40ep_parada_map50/20261028_233314", AZUL, "-"),
    ("FCOS", "VinDR_fcos_40ep_parada_map50/20261107_015816", LARANJA, "-"),
    ("ATSS", "VinDR_atss_40ep_parada_map50/20261107_033803", AQUA, "-"),
    ("Faster R-CNN com recorte (descartado)", "VinDR_faster_rcnn_recorte_40ep_parada_map50/20261105_231743", CINZA, "--"),
]
fig, ax = plt.subplots(figsize=(9, 4.6))
resumo = []
for nome, rel, cor, ls in runs:
    val, _ = ler(rel)
    ep = sorted(val)
    y = [val[e]["coco/bbox_mAP_50"] for e in ep]
    ax.plot(ep, y, ls, color=cor, lw=2, label=nome)
    b = max(ep, key=lambda e: val[e]["coco/bbox_mAP_50"])
    ax.plot([b], [val[b]["coco/bbox_mAP_50"]], "o", ms=8, color=cor, mec=FUNDO, mew=2)
    ax.annotate(f"{pct(val[b]['coco/bbox_mAP_50'])} (ép. {b})", (b, val[b]["coco/bbox_mAP_50"]),
                textcoords="offset points", xytext=(6, -16) if nome == "FCOS" else (6, 6), fontsize=8.5, color=TXT)
    resumo.append((nome, b, val[b]["coco/bbox_mAP_50"], ep[-1]))
ax.set_xlabel("Época")
ax.set_ylabel("mAP@0,50 na validação (92 imagens, 97 massas)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_ylim(0, 0.85)
ax.set_title("VinDr-Mammo: mAP@0,50 de validação por época (ponto = melhor época, checkpoint salvo)",
             loc="left", fontsize=11, color=TXT)
ax.legend(loc="lower right", fontsize=9)
salvar(fig, "g1_vindr_val_map50_por_epoca.png")
print(resumo)

# ---------- G2: 200 épocas sem parada (overfitting) ----------
val, perda = ler("VinDR_faster_rcnn_200ep_sem_parada/20261104_234134")
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
ep = sorted(perda)
a1.plot(ep, [perda[e] for e in ep], color=AZUL, lw=2)
a1.set_ylabel("Loss média de treino")
a1.set_title("Faster R-CNN, 200 épocas sem parada: a loss de treino continua caindo, o mAP de validação não sobe",
             loc="left", fontsize=11, color=TXT)
ev = sorted(val)
a2.plot(ev, [val[e]["coco/bbox_mAP_50"] for e in ev], color=LARANJA, lw=2)
b = max(ev, key=lambda e: val[e]["coco/bbox_mAP_50"])
a2.plot([b], [val[b]["coco/bbox_mAP_50"]], "o", ms=8, color=LARANJA, mec=FUNDO, mew=2)
a2.annotate(f"melhor: {pct(val[b]['coco/bbox_mAP_50'])} na época {b}", (b, val[b]["coco/bbox_mAP_50"]),
            textcoords="offset points", xytext=(8, 4), fontsize=9, color=TXT)
for a in (a1, a2):
    a.axvline(b, color=TXT2, lw=1, ls=":")
a2.set_ylabel("mAP@0,50 (validação)")
a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
a2.set_xlabel("Época")
salvar(fig, "g2_vindr_200ep_overfitting.png")
print("200ep", perda[1], perda[b], perda[ep[-1]], val[ev[-1]]["coco/bbox_mAP_50"])

# ---------- G3: teste oficial ----------
res = json.load(open(os.path.join(RAIZ, "resultados", "resultado_test.json")))["resultados"]
nomes = ["Faster R-CNN", "FCOS", "ATSS", "Ensemble WBF"]
cores = [AZUL, LARANJA, AQUA, AMARELO]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw=dict(width_ratios=[1, 1.1]))
for i, r in enumerate(res):
    a1.bar(i, r["mAP_50"], color=cores[i], width=0.62)
    a1.text(i, r["mAP_50"] + 0.012, pct(r["mAP_50"]), ha="center", fontsize=9, color=TXT)
a1.set_xticks(range(4), nomes, fontsize=9)
a1.set_ylim(0, 0.75)
a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
a1.set_title("mAP@0,50 no teste: mesmo patamar\n(diferenças dentro do ruído, sem IC)", loc="left", fontsize=10.5, color=TXT)
a1.grid(axis="x", visible=False)
for i, r in enumerate(res):
    a2.plot(r["FPPI"], r["TPR"], "o", ms=11, color=cores[i], mec=FUNDO, mew=2)
    a2.annotate(f"{nomes[i]}\nTPR {pct(r['TPR'])} · FPPI {r['FPPI']:.2f}".replace(".", ",", 1) if False else
                f"{nomes[i]}\nTPR {pct(r['TPR'])} · FPPI {fppi_txt(r['FPPI'])}",
                (r["FPPI"], r["TPR"]), textcoords="offset points", xytext=(10, -4), fontsize=8.5, color=TXT)
a2.set_xlim(0, 1.9)
a2.set_ylim(0.4, 0.8)
a2.set_xlabel("FPPI (falsos positivos por imagem) — menor é melhor")
a2.set_ylabel("TPR (sensibilidade)")
a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
a2.set_title("TPR × FPPI (score ≥ 0,3, IoU ≥ 0,5)", loc="left", fontsize=10.5, color=TXT)
fig.suptitle("VinDr-Mammo, teste oficial (219 imagens, 237 massas) — resultados/resultado_test.json",
             x=0.01, ha="left", fontsize=11, color=TXT)
fig.tight_layout()
salvar(fig, "g3_vindr_teste_comparacao.png")

# ---------- G4: CBIS-DDSM, melhor mAP@0,50 por treino ----------
# Fonte: resultados/logs/README.md (treinos válidos D–I e Cascade)
cbis = [("D\n12 ép.", 0.468), ("E\nrepetição de D", 0.455), ("F\nparada mAP", 0.467),
        ("G\nparada loss", 0.445), ("H\nsem parada", 0.451), ("I\nparada mAP@0,50", 0.472),
        ("Cascade\nR-CNN", 0.443)]
vi, _ = ler("cbis_faster_rcnn_40ep_parada_map50/20261017_022145")
vh, _ = ler("cbis_faster_rcnn_40ep_sem_parada/20261016_174247")
print("confere I", max(r["coco/bbox_mAP_50"] for r in vi.values()),
      "H", max(r["coco/bbox_mAP_50"] for r in vh.values()))
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, (n, v) in enumerate(cbis):
    cor = AZUL if n.startswith("I") else CINZA
    ax.bar(i, v, color=cor, width=0.62)
    ax.text(i, v + 0.008, pct(v), ha="center", fontsize=9, color=TXT)
ax.set_xticks(range(len(cbis)), [n for n, _ in cbis], fontsize=8.5)
ax.set_ylim(0, 0.6)
ax.grid(axis="x", visible=False)
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_ylabel("Melhor mAP@0,50 (validação)")
ax.set_title("CBIS-DDSM (247 imagens, 260 massas): mudar o treino não muda o resultado\n"
             "todas as variantes entre 44,3% e 47,2% — D×E (mesma config.) já diferem 1,3 ponto",
             loc="left", fontsize=10.5, color=TXT)
salvar(fig, "g4_cbis_treinos.png")
