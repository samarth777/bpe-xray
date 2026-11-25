#!/bin/bash
# usage: bsub -q normal -n 4 -M 48G -R "rusage[mem=48G]" -gpu "num=1:mode=exclusive_process:gmem=40G" -o logs/x.%J.log ./job.sh src/script.py [args...]
set -euo pipefail
set -x
S=/tmp/bpe-xray-$USER
mkdir -p "$S"
export HF_HOME="$S/hf"
export HF_XET_HIGH_PERFORMANCE=1
export OMP_NUM_THREADS=8
export UV_CACHE_DIR="$S/uvc"
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$S/venv"
export TOKENIZERS_PARALLELISM=false

hostname
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

cd "$HOME/bpe-xray"
# Prefer cu128 venv (works across ALL A100/H100 driver versions), else cu132 venv
PY=""
for cand in "$HOME/raysense/.venv/bin/python" "$HOME/loop-lens/.venv/bin/python"; do
    if [ -x "$cand" ] && "$cand" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        PY="$cand"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "ERROR: No working CUDA Python found on $(hostname)!" >&2
    exit 1
fi
echo "Selected Python: $PY"
export PYTHONPATH="$HOME/bpe-xray:$HOME/bpe-xray/src:${PYTHONPATH:-}"
exec "$PY" -u "$@"
