"""
metricas.py - sensibilidade (TPR) e falsos positivos por imagem (FPPI)

Usado pelo VinDR_inference.py e pelo VinDR_ensemble.py, para que os dois
calculem do mesmo jeito (o do ensemble, que gerou a tabela do README).

Pareamento 1 para 1: em cada imagem, as deteccoes com score >= thr sao
percorridas da maior para a menor confianca; cada uma fica com a massa ainda
livre de maior IoU. Se esse IoU for >= iou_min, a massa conta como achada
(e nao pode ser usada de novo); senao, a deteccao e um falso positivo.
Assim, duas caixas na mesma massa contam como 1 acerto + 1 falso positivo.
"""

# Score minimo padrao para TPR/FPPI e para exibir deteccoes: usado pela CLI
# (VinDR_inference.py --thr), pelo VinDR_ensemble.py --thr e como valor inicial
# do controle deslizante da interface (via /status do main.py).
THR_PADRAO = 0.3


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def tpr_fppi(gt, det, thr, iou_min=0.5):
    """Sensibilidade e falsos positivos por imagem com score >= thr (cada massa conta uma vez).

    gt:  {image_id: [[x1, y1, x2, y2], ...]}   todas as imagens avaliadas (lista vazia = sem massa)
    det: {image_id: [[x1, y1, x2, y2, score], ...]}
    O FPPI e dividido pelo numero de imagens em gt.
    """
    achadas = total = fp = 0
    for iid, caixas_gt in gt.items():
        dets = sorted([d for d in det.get(iid, []) if d[4] >= thr], key=lambda d: -d[4])
        usadas = set()
        for d in dets:
            melhor, k_melhor = 0.0, -1
            for k, g in enumerate(caixas_gt):
                if k not in usadas:
                    v = iou(d, g)
                    if v > melhor:
                        melhor, k_melhor = v, k
            if melhor >= iou_min:
                usadas.add(k_melhor)
            else:
                fp += 1
        achadas += len(usadas)
        total += len(caixas_gt)
    return achadas / max(total, 1), fp / max(len(gt), 1)
