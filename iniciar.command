#!/bin/bash
# iniciar.command - liga a interface web do TCC no macOS (duplo clique).
# Faz o mesmo que o iniciar.bat do Windows.

cd "$(dirname "$0")" || exit 1
export PYTHONIOENCODING=utf-8
pausar() { echo; read -r -p "Pressione Enter para fechar esta janela..." _; }

echo
echo "================================================================"
echo "  Detecção de Massas em Mamografia - iniciando"
echo "================================================================"
echo

# ---------------------------------------------------------------- 1. conda
echo "[1/4] Procurando o conda..."
CONDA_ROOT=""
command -v conda >/dev/null 2>&1 && CONDA_ROOT="$(conda info --base 2>/dev/null)"
[ -z "$CONDA_ROOT" ] && command -v mamba >/dev/null 2>&1 && CONDA_ROOT="$(mamba info --base 2>/dev/null)"
for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" \
         /opt/homebrew/Caskroom/miniforge/base /opt/miniconda3 /opt/anaconda3 /usr/local/anaconda3; do
    [ -z "$CONDA_ROOT" ] && [ -f "$d/etc/profile.d/conda.sh" ] && CONDA_ROOT="$d"
done

PY=python3
if [ -n "$CONDA_ROOT" ] && [ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]; then
    echo "      Conda em: $CONDA_ROOT"
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"

    AMBIENTE=""
    [ -f ambiente_local.txt ] && AMBIENTE="$(head -n 1 ambiente_local.txt)"
    if [ -z "$AMBIENTE" ]; then
        achados=()
        for e in "$CONDA_ROOT" "$CONDA_ROOT"/envs/* "$HOME"/.conda/envs/*; do
            compgen -G "$e/lib/python3*/site-packages/mmdet*" >/dev/null && achados+=("$e")
        done
        echo "      Ambientes com o mmdet instalado: ${#achados[@]}"
        for e in "${achados[@]}"; do echo "        - $e"; done
        [ ${#achados[@]} -eq 1 ] && AMBIENTE="${achados[0]}"
    fi

    while true; do
        if [ -z "$AMBIENTE" ]; then
            echo
            echo "  Digite o NOME do ambiente conda usado no treino (o de 'conda activate NOME')."
            echo "  A resposta fica salva em ambiente_local.txt."
            read -r -p "  Nome do ambiente: " AMBIENTE
            [ -z "$AMBIENTE" ] && continue
            echo "$AMBIENTE" > ambiente_local.txt
        fi
        echo "      Ativando o ambiente: $AMBIENTE"
        if conda activate "$AMBIENTE"; then break; fi
        echo "  ❌ Não consegui ativar o ambiente \"$AMBIENTE\"."
        rm -f ambiente_local.txt
        AMBIENTE=""
    done
    PY=python
else
    echo "      Não encontrei o conda. Vou usar o $PY do sistema."
    command -v "$PY" >/dev/null 2>&1 || { echo "  ❌ Python não encontrado."; pausar; exit 1; }
fi

# ---------------------------------------------------------------- 2-3. bibliotecas e verificação
echo
echo "[2/4] Conferindo as bibliotecas da interface web..."
"$PY" -m pip install --disable-pip-version-check -q -r requirements-web.txt \
    || echo "      Não consegui instalar agora (sem internet?). O verificador mostra o que falta."

echo
echo "[3/4] Verificando se está tudo pronto..."
"$PY" verificar.py
case $? in
    2) echo "  O servidor não pode ser ligado. Corrija os itens com ❌ acima."; pausar; exit 1 ;;
    1) echo "  ATENÇÃO: vai abrir em MODO DEMONSTRAÇÃO. As caixas serão FALSAS."
       read -r -p "  Enter para continuar mesmo assim (ou feche a janela para corrigir antes)... " _ ;;
esac

# ---------------------------------------------------------------- 4. servidor + navegador
if curl -s -o /dev/null --max-time 2 http://localhost:8000/status; then
    echo
    echo "  O servidor já está ligado (talvez em outra janela). Abrindo o navegador..."
    open http://localhost:8000
    pausar
    exit 0
fi

# Abre o navegador sozinho assim que o servidor responder (o modelo pode demorar a carregar)
(
    for _ in $(seq 300); do
        if curl -s -o /dev/null --max-time 2 http://localhost:8000/status; then
            open http://localhost:8000
            break
        fi
        sleep 1
    done
) &

echo
echo "[4/4] Ligando o servidor..."
echo "================================================================"
echo "  O navegador abre sozinho quando o modelo terminar de carregar."
echo "  Endereço: http://localhost:8000"
echo
echo "  FECHE ESTA JANELA PARA DESLIGAR O SERVIDOR (ou Ctrl+C)."
echo "================================================================"
echo
"$PY" -m uvicorn main:app --port 8000
echo
echo "  O servidor parou. Veja a mensagem acima."
pausar
