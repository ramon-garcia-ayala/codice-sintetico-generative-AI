"""Métricas por imagen y reglas de descarte.

Los umbrales de `FilterConfig` están calibrados contra las 282 imágenes que ya
existían en `02_DATASET`. La auditoría de ese conjunto es el test de regresión
del módulo: mover un umbral debe cambiar los conteos esperados en
`tests/test_filters.py`, no romperlos por accidente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .hashing import laplacian_variance, load_thumb, phash
from .models import RECOMPUTED_REJECT_REASONS, ImageRecord, RejectReason

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}


@dataclass(frozen=True)
class FilterConfig:
    """Umbrales de descarte. Todos configurables por CLI."""

    #: SDXL se entrena a 1024. Subir de resolución no añade información.
    min_short_side: int = 1024

    #: A partir de aquí la imagen va al bucket panorámico, que alimenta la
    #: sección madre de 3072x1024 en vez del entrenamiento cuadrado.
    panoramic_aspect: float = 2.0

    #: Más allá de esto es una tira inservible incluso como panorámica.
    max_aspect: float = 3.5

    #: Fracción del lado menor que se considera "marco" para estimar el fondo.
    border_frac: float = 0.12

    #: Un marco muy uniforme (std baja) y muy oscuro o muy claro delata una
    #: foto de estudio o de vitrina de museo: fondo infinito, no yacimiento.
    border_std_max: float = 32.0
    border_dark_mean_max: float = 62.0
    border_light_mean_min: float = 205.0

    #: O bien la imagen es mayoritariamente negro o blanco puro.
    dark_pixel_frac_max: float = 0.40
    light_pixel_frac_max: float = 0.32

    #: Umbrales de intensidad para contar píxel "casi negro" / "casi blanco".
    dark_level: int = 30
    light_level: int = 225

    #: Varianza del laplaciano sobre la miniatura de 256 px. Conservador a
    #: propósito: descartar roca de textura fina por "borrosa" sería peor que
    #: dejar pasar alguna foto blanda.
    min_sharpness: float = 40.0

    #: Distancia de Hamming bajo la cual dos pHash se consideran la misma imagen.
    dup_hamming_max: int = 5


@dataclass
class ImageMetrics:
    """Lo que se mide de una imagen, antes de decidir nada sobre ella."""

    width: int
    height: int
    mode: str
    bytes: int
    phash: str
    sharpness: float
    border_mean: float
    border_std: float
    dark_frac: float
    light_frac: float
    saturation: float

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    @property
    def aspect(self) -> float:
        return max(self.width, self.height) / min(self.width, self.height)


def measure(path: Path, cfg: FilterConfig | None = None) -> ImageMetrics:
    """Mide una imagen. Lee dimensiones del original y el resto del thumbnail."""
    cfg = cfg or FilterConfig()

    with Image.open(path) as probe:
        width, height = probe.size
        mode = probe.mode

    thumb = load_thumb(path)
    a = np.asarray(thumb, dtype=np.float32)
    gray = a.mean(axis=2)

    h, w = gray.shape
    m = max(2, int(min(h, w) * cfg.border_frac))
    border = np.concatenate(
        [gray[:m].ravel(), gray[-m:].ravel(), gray[:, :m].ravel(), gray[:, -m:].ravel()]
    )

    return ImageMetrics(
        width=width,
        height=height,
        mode=mode,
        bytes=path.stat().st_size,
        phash=phash(thumb),
        sharpness=laplacian_variance(thumb),
        border_mean=float(border.mean()),
        border_std=float(border.std()),
        dark_frac=float((gray < cfg.dark_level).mean()),
        light_frac=float((gray > cfg.light_level).mean()),
        saturation=float((a.max(axis=2) - a.min(axis=2)).mean()),
    )


def is_studio_background(m: ImageMetrics, cfg: FilterConfig | None = None) -> bool:
    """True si la imagen parece foto de estudio o vitrina en vez de campo.

    Dos señales independientes, cualquiera basta:

    1. El marco perimetral es uniforme y extremo — fondo infinito de estudio.
    2. La imagen es mayoritariamente negro o blanco puro.

    Detecta, entre otras, las fotos de mineral de colección con cartela
    museográfica que contaminaban el dataset original.
    """
    cfg = cfg or FilterConfig()
    uniform_backdrop = m.border_std < cfg.border_std_max and (
        m.border_mean < cfg.border_dark_mean_max or m.border_mean > cfg.border_light_mean_min
    )
    blown_out = m.dark_frac > cfg.dark_pixel_frac_max or m.light_frac > cfg.light_pixel_frac_max
    return uniform_backdrop or blown_out


def apply_filters(
    rec: ImageRecord, m: ImageMetrics, cfg: FilterConfig | None = None
) -> ImageRecord:
    """Escribe las métricas en el record y **recalcula** los descartes de píxel.

    No borra archivos: sólo marca `rejected` con sus razones, para que el
    contact sheet muestre lo descartado y permita revertirlo.

    Las razones de `RECOMPUTED_REJECT_REASONS` se limpian antes de reevaluar.
    Sin eso, reejecutar `audit` con un umbral más bajo no tendría ningún efecto
    visible —el veredicto viejo seguiría pegado al record— y además desharía
    silenciosamente cualquier rescate hecho con `refilter`. Las razones que no
    dependen de los píxeles (`MANUAL`, `LICENSE_DENIED`, `OUT_OF_SCOPE`) y el
    `DUPLICATE` que calcula `find_duplicates` en otra pasada sobreviven intactas.
    """
    cfg = cfg or FilterConfig()

    rec.width, rec.height = m.width, m.height
    rec.mode, rec.bytes = m.mode, m.bytes
    rec.phash, rec.sharpness = m.phash, m.sharpness
    rec.saturation = m.saturation

    rec.reject_reasons = [
        r for r in rec.reject_reasons if r not in RECOMPUTED_REJECT_REASONS
    ]
    # `panoramic` también es un veredicto derivado de las métricas: si no se
    # reinicia, una imagen deja de ser panorámica sólo cuando alguien lo edita
    # a mano.
    rec.panoramic = False

    if m.short_side < cfg.min_short_side:
        rec.reject(RejectReason.TOO_SMALL)

    if is_studio_background(m, cfg):
        rec.reject(RejectReason.STUDIO_BACKGROUND)

    if m.sharpness < cfg.min_sharpness:
        rec.reject(RejectReason.BLURRY)

    if m.aspect > cfg.max_aspect:
        rec.reject(RejectReason.EXTREME_ASPECT)
    elif m.aspect >= cfg.panoramic_aspect:
        # No es un defecto: la sección madre del brief es 3:1.
        rec.panoramic = True

    if m.mode not in ("RGB", "L"):
        note = f"convertir a RGB (mode={m.mode})"
        if note not in rec.needs_review:
            rec.needs_review.append(note)

    rec.rejected = bool(rec.reject_reasons)
    return rec


def find_duplicates(
    records: list[ImageRecord], cfg: FilterConfig | None = None
) -> dict[str, str]:
    """Mapea filename duplicado -> filename conservado.

    Primero agrupa por `sha256` (idénticos byte a byte), luego compara `phash`
    por pares para los casi-idénticos: la misma foto reescalada o recomprimida,
    que es el solape típico entre Wikimedia y las fuentes institucionales.

    O(n^2) sobre los pHash. Con unos pocos miles de imágenes es irrelevante.
    """
    cfg = cfg or FilterConfig()
    dupes: dict[str, str] = {}

    by_sha: dict[str, str] = {}
    for rec in records:
        if not rec.sha256:
            continue
        if rec.sha256 in by_sha:
            dupes[rec.filename] = by_sha[rec.sha256]
        else:
            by_sha[rec.sha256] = rec.filename

    kept: list[ImageRecord] = []
    for rec in records:
        if not rec.phash or rec.filename in dupes:
            continue
        match = next(
            (k for k in kept if hamming_le(rec.phash, k.phash, cfg.dup_hamming_max)), None
        )
        if match:
            dupes[rec.filename] = match.filename
        else:
            kept.append(rec)

    return dupes


def reevaluate_min_short_side(rec: ImageRecord, min_short_side: int) -> bool:
    """Reaplica sólo el umbral de resolución sobre métricas ya guardadas.

    Existe para casos como el plastiglomerado real: 186 imágenes ya medidas y
    descartadas casi todas por `too_small` contra el umbral global de 1024 px,
    cuando la clase es tan escasa que 640 px (el mínimo de bucket que ya usa
    kohya) es un piso razonable sólo para ella. No hace falta re-descargar ni
    re-medir nada — el ancho y alto ya están en el manifest.

    Deliberadamente sólo toca `TOO_SMALL`: otras razones de descarte (fondo de
    estudio, duplicado…) no dependen de la resolución y no deben tocarse aquí.
    Devuelve True si el veredicto de la imagen cambió.
    """
    if rec.short_side is None:
        return False

    had_it = RejectReason.TOO_SMALL in rec.reject_reasons
    should_have_it = rec.short_side < min_short_side

    if had_it == should_have_it:
        return False

    if should_have_it:
        rec.reject(RejectReason.TOO_SMALL)
    else:
        rec.reject_reasons.remove(RejectReason.TOO_SMALL)
        rec.rejected = bool(rec.reject_reasons)

    return True


def hamming_le(a: str | None, b: str | None, threshold: int) -> bool:
    """Comparación de pHash tolerante a valores ausentes."""
    if not a or not b:
        return False
    from .hashing import hamming

    return hamming(a, b) <= threshold


def iter_images(root: Path) -> list[Path]:
    """Imágenes en `root`, ordenadas. Ignora `desktop.ini` y demás basura."""
    return sorted(
        p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
