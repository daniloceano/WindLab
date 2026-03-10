#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Configuração do ambiente conda para o WindLab
# TC nº 050.0125966.23.9 — PETROBRAS/CENPES & USP/IAG
#
# Uso:
#   bash setup_env.sh              # cria/atualiza o ambiente 'windlab'
#   bash setup_env.sh tc_petrobras # instala dependências em ambiente existente
#
# O script:
#   1. Verifica se o conda está disponível.
#   2. Cria o ambiente 'windlab' a partir de environment.yml
#      (ou atualiza, se já existir).
#   3. Instala o WindLab em modo editável (pip install -e .).
#   4. Exibe instruções de ativação.
# =============================================================================
set -euo pipefail

# ── Configuração ──────────────────────────────────────────────────────────────
ENV_NAME="${1:-windlab}"                        # nome do ambiente (padrão: windlab)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # raiz do repositório
ENV_FILE="$REPO_DIR/environment.yml"

# ── Verificações iniciais ─────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "❌  conda não encontrado. Instale o Miniconda ou Anaconda e tente novamente."
    echo "    https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "╔══════════════════════════════════════════════════════════╗"
echo "║          WindLab — Configuração do Ambiente conda        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo
echo "  Repositório : $REPO_DIR"
echo "  Ambiente    : $ENV_NAME"
echo

# ── Criar ou atualizar o ambiente ─────────────────────────────────────────────
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "▶  Ambiente '$ENV_NAME' já existe. Atualizando dependências…"
    conda env update --name "$ENV_NAME" --file "$ENV_FILE" --prune
else
    echo "▶  Criando ambiente '$ENV_NAME' a partir de environment.yml…"
    # Remove a linha 'windlab' do pip install temporariamente para instalar em modo editável
    conda env create --name "$ENV_NAME" --file "$ENV_FILE" || true
fi

# ── Instalar WindLab em modo editável ─────────────────────────────────────────
echo
echo "▶  Instalando WindLab em modo editável (pip install -e .)…"
conda run --no-capture-output -n "$ENV_NAME" pip install -e "$REPO_DIR"

# ── Verificar instalação ──────────────────────────────────────────────────────
echo
echo "▶  Verificando instalação…"
conda run --no-capture-output -n "$ENV_NAME" python - <<'EOF'
import windlab
from windlab.ingestion import CustomDataReader, PRESET_NAMES, AVAILABLE_PRESETS
print(f"  ✓ windlab importado com sucesso")
print(f"  ✓ Presets disponíveis: {AVAILABLE_PRESETS}")
EOF

# ── Instruções finais ─────────────────────────────────────────────────────────
echo
echo "══════════════════════════════════════════════════════════════"
echo "  ✅  Ambiente '$ENV_NAME' configurado com sucesso!"
echo
echo "  Para ativar:"
echo "    conda activate $ENV_NAME"
echo
echo "  Para iniciar a interface gráfica:"
echo "    windlab-gui"
echo
echo "  Para executar um script de exemplo:"
echo "    python examples/WindLab_example_usage.py"
echo "══════════════════════════════════════════════════════════════"
