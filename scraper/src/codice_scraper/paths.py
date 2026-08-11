"""Rutas del proyecto.

El Drive compartido de Carlos se monta como unidad de red. Está declarado aquí
como constante para que quede explícito y en un solo sitio: **es de sólo
lectura para este paquete**. Copiamos a `dataset/_incoming/` y trabajamos ahí;
una escritura in-place se sincronizaría a todo el equipo.
"""

from __future__ import annotations

import os
from pathlib import Path

from .models import ImageClass

#: Raíz del repositorio (…/scraper/src/codice_scraper/paths.py -> 3 niveles arriba de src).
REPO_ROOT = Path(__file__).resolve().parents[3]

DATASET_DIR = REPO_ROOT / "dataset"
INCOMING_DIR = DATASET_DIR / "_incoming"
REJECTED_DIR = DATASET_DIR / "_rejected"
MANIFEST_PATH = DATASET_DIR / "manifest.jsonl"
ATTRIBUTIONS_PATH = DATASET_DIR / "ATTRIBUTIONS.md"
REPORTS_DIR = REPO_ROOT / "reports"

#: Drive de Carlos montado localmente. Sobreescribible con CODICE_DRIVE.
DRIVE_ROOT = Path(
    os.environ.get(
        "CODICE_DRIVE",
        r"G:\.shortcut-targets-by-id\1mc34MerUqYC9EHfscYh8sNqiX6S4KPic\CODICE SIN",
    )
)

DRIVE_DATASET = DRIVE_ROOT / "02_DATASET"
DRIVE_LORA = DRIVE_ROOT / "03_LORA"
DRIVE_ESCANEO = DRIVE_ROOT / "01_ESCANEO"


def class_dir(klass: ImageClass) -> Path:
    return DATASET_DIR / klass.value


def ensure_dirs() -> None:
    """Crea el árbol de `dataset/`. No toca el Drive."""
    for path in (DATASET_DIR, INCOMING_DIR, REJECTED_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for klass in ImageClass:
        class_dir(klass).mkdir(parents=True, exist_ok=True)
