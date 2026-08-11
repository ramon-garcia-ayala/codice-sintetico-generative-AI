#!/usr/bin/env bash
# Prepara un pod de RunPod para entrenar el LoRA de Códice Sintético.
#
# Todo vive bajo /workspace, que es donde RunPod monta el Network Volume: así
# el checkpoint de SDXL (~7 GB) y el dataset sobreviven al apagado del pod y no
# hay que volver a descargarlos. Apagar el pod deja de cobrar cómputo; el
# volumen cuesta aparte y es barato.
#
#   bash bootstrap.sh
#
# Idempotente: se puede volver a correr sin romper nada.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
KOHYA_DIR="$WORKSPACE/sd-scripts"
MODELS_DIR="$WORKSPACE/models"
VENV="$WORKSPACE/venv"

# Commit fijado. sd-scripts rompe compatibilidad de flags entre versiones con
# cierta frecuencia; si esto flota, un día el TOML deja de parsear sin aviso.
KOHYA_REF="${KOHYA_REF:-sd3}"

echo "==> Directorios"
mkdir -p "$MODELS_DIR" "$WORKSPACE/dataset" "$WORKSPACE/output" "$WORKSPACE/logs"

echo "==> Dependencias del sistema"
if ! command -v git >/dev/null; then
    apt-get update -qq && apt-get install -y -qq git wget aria2
fi

echo "==> kohya_ss / sd-scripts"
if [ ! -d "$KOHYA_DIR/.git" ]; then
    git clone --depth 1 --branch "$KOHYA_REF" \
        https://github.com/kohya-ss/sd-scripts.git "$KOHYA_DIR"
else
    echo "    ya presente, se omite el clone"
fi

echo "==> Entorno virtual"
if [ ! -d "$VENV" ]; then
    python -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip -q

echo "==> Paquetes de Python (unos minutos la primera vez)"
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -q -r "$KOHYA_DIR/requirements.txt"
# bitsandbytes aporta AdamW8bit, que es lo que permite entrenar SDXL a 1024
# dentro de los 24 GB de una 4090 sin recortar el batch a menos de 1.
pip install -q bitsandbytes

echo "==> Modelos base"
download() {  # url destino
    if [ -f "$2" ]; then
        echo "    ya está: $(basename "$2")"
        return
    fi
    echo "    bajando $(basename "$2")"
    aria2c -x 8 -s 8 --console-log-level=warn -d "$(dirname "$2")" \
        -o "$(basename "$2")" "$1"
}

download \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors" \
    "$MODELS_DIR/sd_xl_base_1.0.safetensors"

# El VAE de SDXL produce NaN en fp16. Este fork corrige ese fallo y es el mismo
# que ya usan los notebooks del repo (`--pretrained_vae_model_name_or_path`).
download \
    "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors" \
    "$MODELS_DIR/sdxl_vae_fp16_fix.safetensors"

echo "==> accelerate"
mkdir -p ~/.cache/huggingface/accelerate
cat > ~/.cache/huggingface/accelerate/default_config.yaml <<'YAML'
compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
YAML

echo ""
echo "Listo."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""
echo "Siguiente:"
echo "  1. Sube el dataset:  bash sync.sh pull"
echo "  2. Entrena:          bash train.sh"
