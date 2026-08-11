#!/usr/bin/env bash
# Mueve dataset y pesos entre el pod y el Drive del proyecto vía rclone.
#
#   bash sync.sh setup    configura el remoto de Drive (una vez por volumen)
#   bash sync.sh pull     Drive -> pod   (trae dataset/train)
#   bash sync.sh push     pod -> Drive   (sube los .safetensors a 03_LORA)
#
# El remoto se llama `gdrive` y apunta a la carpeta compartida "CODICE SIN".
# Como el Network Volume persiste, `setup` sólo hace falta la primera vez.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REMOTE="${REMOTE:-gdrive}"
DRIVE_ROOT="${DRIVE_ROOT:-CODICE SIN}"
ACTION="${1:-}"

need_rclone() {
    if ! command -v rclone >/dev/null; then
        echo "==> Instalando rclone"
        curl -fsSL https://rclone.org/install.sh | bash
    fi
}

case "$ACTION" in
setup)
    need_rclone
    echo "Se abrirá la configuración interactiva de rclone."
    echo ""
    echo "  1. n  (new remote)"
    echo "  2. nombre: $REMOTE"
    echo "  3. storage: drive"
    echo "  4. client_id y client_secret: vacíos"
    echo "  5. scope: 1  (acceso completo)"
    echo "  6. Edit advanced config: n"
    echo "  7. Use auto config: **n**  <- importante, el pod no tiene navegador"
    echo "     Pega en tu máquina el comando que te muestre y devuelve el token."
    echo ""
    rclone config
    ;;

pull)
    need_rclone
    echo "==> Trayendo dataset desde Drive"
    mkdir -p "$WORKSPACE/dataset"
    rclone copy "$REMOTE:$DRIVE_ROOT/02_DATASET/train" "$WORKSPACE/dataset/train" \
        --progress --transfers 8 --checkers 16
    echo ""
    find "$WORKSPACE/dataset/train" -maxdepth 1 -type d -not -path "*/train" \
        -exec sh -c 'echo "    $(basename "$1"): $(find "$1" -name "*.jpg" | wc -l) imgs"' _ {} \;
    ;;

push)
    need_rclone
    if ! ls "$WORKSPACE/output"/*.safetensors >/dev/null 2>&1; then
        echo "error: no hay .safetensors en $WORKSPACE/output" >&2
        exit 1
    fi
    echo "==> Subiendo pesos a Drive (03_LORA)"
    rclone copy "$WORKSPACE/output" "$REMOTE:$DRIVE_ROOT/03_LORA" \
        --include "*.safetensors" --progress
    # Las muestras dejan ver la deriva del entrenamiento sin bajar los pesos.
    if [ -d "$WORKSPACE/output/sample" ]; then
        rclone copy "$WORKSPACE/output/sample" "$REMOTE:$DRIVE_ROOT/03_LORA/samples" --progress
    fi
    echo "Listo."
    ;;

*)
    sed -n '2,12p' "$0"
    exit 1
    ;;
esac
