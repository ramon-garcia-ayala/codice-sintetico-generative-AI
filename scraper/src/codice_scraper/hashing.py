"""Hashes perceptuales y métricas de imagen, en numpy puro.

No dependemos de `imagehash` ni de OpenCV a propósito: lo único que hace falta
de ellos aquí son una DCT-II de 32x32 y una convolución laplaciana 3x3, y ambas
caben en unas cuantas líneas. Menos dependencias en un entorno conda compartido.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

# Tamaño al que se reduce la imagen antes de la DCT, y lado del bloque de
# coeficientes de baja frecuencia que forma el hash. 32/8 son los valores
# clásicos de pHash y dan 64 bits.
PHASH_IMG_SIZE = 32
PHASH_BLOCK = 8

# Lado máximo de la miniatura de trabajo. Todas las métricas de `filters` se
# calculan sobre esta reducción, así que el valor forma parte del contrato:
# cambiarlo mueve los umbrales y rompe la reproducibilidad de la auditoría.
THUMB_SIZE = 256


@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    """Matriz de la DCT-II sin normalizar: `D @ A @ D.T` es la DCT-II 2D de A.

    Sin normalizar basta porque pHash solo compara coeficientes contra su
    propia mediana; cualquier factor de escala común se cancela.
    """
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    return np.cos(np.pi * (x + 0.5) * k / n)


def phash(img: Image.Image) -> str:
    """pHash de 64 bits como 16 dígitos hexadecimales."""
    g = img.convert("L").resize((PHASH_IMG_SIZE, PHASH_IMG_SIZE), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.float64)

    d = _dct_matrix(PHASH_IMG_SIZE)
    coeffs = d @ a @ d.T

    block = coeffs[:PHASH_BLOCK, :PHASH_BLOCK].flatten()
    # Se excluye el término DC del cálculo de la mediana: domina en magnitud y
    # sesgaría el umbral hacia un lado, dejando el hash casi sin información.
    med = np.median(block[1:])
    bits = block > med

    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    """Distancia de Hamming entre dos pHash hexadecimales."""
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def laplacian_variance(img: Image.Image) -> float:
    """Varianza del laplaciano: proxy de nitidez. Valores bajos = desenfoque.

    Kernel [[0,1,0],[1,-4,1],[0,1,0]] aplicado por slicing, sin scipy.
    """
    g = np.asarray(img.convert("L"), dtype=np.float64)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
        - 4.0 * g[1:-1, 1:-1]
    )
    return float(lap.var())


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 del archivo, leído por bloques para no cargar 4 MB de golpe."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while data := fh.read(chunk):
            h.update(data)
    return h.hexdigest()


def load_thumb(path_or_img: Path | str | Image.Image, size: int = THUMB_SIZE) -> Image.Image:
    """Carga una imagen reducida a `size` px de lado mayor, en RGB.

    Usa `draft()` antes de decodificar, que en JPEG permite al decodificador
    saltar niveles de la DCT y devolver directamente una versión reducida. Con
    imágenes de 5000 px sobre Google Drive montado la diferencia es de minutos.

    Este pipeline exacto (draft -> convert -> thumbnail) es parte del contrato
    de reproducibilidad de la auditoría: las métricas de `filters` se calibraron
    sobre él.
    """
    if isinstance(path_or_img, Image.Image):
        img = path_or_img.copy()
    else:
        img = Image.open(path_or_img)
        img.draft("RGB", (size, size))
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    return img
