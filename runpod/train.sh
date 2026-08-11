#!/usr/bin/env bash
# Entrena el LoRA `codice_geo`.
#
#   bash train.sh                 corrida completa
#   bash train.sh --smoke         10 pasos, para validar que el árbol de datos
#                                 parsea antes de comprometer una hora de GPU
#   VERSION=v2 bash train.sh      cambia el nombre de salida

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
KOHYA_DIR="$WORKSPACE/sd-scripts"
CONFIG="${CONFIG:-$(dirname "$0")/train_codice_geo.toml}"
VERSION="${VERSION:-v1}"
DATASET="$WORKSPACE/dataset/train"

# shellcheck disable=SC1091
source "$WORKSPACE/venv/bin/activate"

if [ ! -d "$DATASET" ]; then
    echo "error: no hay dataset en $DATASET" >&2
    echo "       corre 'bash sync.sh pull' primero" >&2
    exit 1
fi

echo "==> Dataset"
total=0
for dir in "$DATASET"/*/; do
    [ -d "$dir" ] || continue
    name=$(basename "$dir")
    imgs=$(find "$dir" -maxdepth 1 -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
    txts=$(find "$dir" -maxdepth 1 -type f -name '*.txt' | wc -l)
    repeats="${name%%_*}"
    echo "    $name: $imgs imgs, $txts captions, x$repeats"
    if [ "$imgs" -ne "$txts" ]; then
        echo "    ATENCION: faltan captions en $name; kohya entrenaria con caption vacio" >&2
    fi
    total=$((total + imgs * repeats))
done
echo "    -> $total vistas por epoca"

EXTRA=()
if [ "${1:-}" = "--smoke" ]; then
    echo ""
    echo "==> Prueba de humo: 10 pasos"
    EXTRA=(--max_train_steps=10 --save_every_n_epochs=999 --output_name="smoke_test")
else
    EXTRA=(--output_name="codice_geo_$VERSION")
fi

# Prompts de muestra. El primero mide el concepto en sí; el segundo prueba que
# el trigger se pueda combinar con un contexto que no está en el dataset, que
# es justamente lo que hará la fricción del proyecto.
SAMPLES="$WORKSPACE/dataset/sample_prompts.txt"
if [ ! -f "$SAMPLES" ]; then
    cat > "$SAMPLES" <<'EOF'
codice_geo, sedimentary strata, laminated sandstone outcrop, horizontal bedding, close detail --n blurry, text, watermark --w 1024 --h 1024 --d 42 --s 28 --l 7
codice_geo, plastiglomerate, fused plastic and sediment, heterogeneous matrix --n blurry, text, watermark --w 1024 --h 1024 --d 42 --s 28 --l 7
codice_geo, sedimentary strata, wide panoramic section, layered strata --n blurry, text --w 1536 --h 512 --d 42 --s 28 --l 7
EOF
fi

cd "$KOHYA_DIR"
echo ""
echo "==> Entrenando"
accelerate launch \
    --num_cpu_threads_per_process 4 \
    sdxl_train_network.py \
    --config_file "$CONFIG" \
    "${EXTRA[@]}"

echo ""
echo "==> Listo. Salida en $WORKSPACE/output"
ls -lh "$WORKSPACE/output"/*.safetensors 2>/dev/null || true
echo ""
echo "Para traerlo a Drive:  bash sync.sh push"
