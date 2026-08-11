"""Fuentes de imágenes.

Cada una implementa `Source.search()` y es independiente de las demás: si una
cambia de API o su licencia resulta no permitir entrenamiento, se desactiva sin
tocar el resto del paquete.
"""

from __future__ import annotations

from .base import Source, http_session
from .europepmc import EuropePMCSource
from .flickr import FlickrSource
from .wikimedia import WikimediaSource

#: Registro de fuentes por nombre.
SOURCES: dict[str, type[Source]] = {
    WikimediaSource.name: WikimediaSource,
    EuropePMCSource.name: EuropePMCSource,
    FlickrSource.name: FlickrSource,
}

#: Las que corren sin credenciales ni decisiones pendientes sobre licencia.
DEFAULT_SOURCES = ("wikimedia", "europepmc")

__all__ = [
    "SOURCES",
    "DEFAULT_SOURCES",
    "Source",
    "WikimediaSource",
    "EuropePMCSource",
    "FlickrSource",
    "http_session",
]


def build_source(name: str, config: dict | None = None) -> Source:
    """Instancia una fuente por nombre."""
    if name not in SOURCES:
        raise KeyError(f"fuente desconocida: {name!r}. Disponibles: {sorted(SOURCES)}")
    return SOURCES[name](config)
