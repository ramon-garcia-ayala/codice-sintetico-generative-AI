"""Wikimedia Commons vía la API de MediaWiki.

Fuente principal del proyecto: sin clave, sin captcha, con licencias explícitas
en los metadatos y cobertura amplia de geología. Es también la única que
permite reconstruir la procedencia del dataset heredado (ver `recover.py`).

Dos modos de descubrimiento, porque se complementan:

- **categoría** — precisa pero superficial: `categorymembers` devuelve sólo los
  miembros directos, y Commons anida mucho en subcategorías.
- **búsqueda** — cubre lo que las categorías dejan fuera, a cambio de más ruido.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from ..models import ImageClass, ImageRecord
from .base import Source, http_session

API = "https://commons.wikimedia.org/w/api.php"

#: Extensiones que sirven para entrenar. Commons aloja mucho SVG, PDF y TIFF de
#: mapas y diagramas que no son fotografía.
USABLE_EXT = (".jpg", ".jpeg", ".png")

#: Por debajo de esto no vale la pena ni descargar: el filtro de resolución la
#: descartaría después.
MIN_WIDTH = 1024


def _strip_html(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _extract(meta: dict, key: str) -> str | None:
    return _strip_html((meta.get(key) or {}).get("value"))


def record_from_page(page: dict, klass: ImageClass, hint: str) -> ImageRecord | None:
    """Convierte una página de la API en `ImageRecord`, o None si no sirve."""
    info = (page.get("imageinfo") or [{}])[0]
    url = info.get("url")
    title = page.get("title", "")

    if not url or not title.lower().endswith(USABLE_EXT):
        return None
    if (info.get("width") or 0) < MIN_WIDTH or (info.get("height") or 0) < MIN_WIDTH:
        return None

    meta = info.get("extmetadata") or {}
    categories = [
        c.get("title", "").removeprefix("Category:")
        for c in (page.get("categories") or [])
    ]
    if hint not in categories:
        categories.append(hint)

    # `File:Nombre con espacios.jpg` -> `wm_nombre_con_espacios.jpg`
    stem = title.removeprefix("File:")
    safe = re.sub(r"[^\w.-]+", "_", stem).strip("_").lower()

    return ImageRecord(
        filename=f"wm_{safe}",
        source="wikimedia",
        source_id=str(page.get("pageid", "")),
        origin_title=title,
        origin_url=info.get("descriptionurl"),
        download_url=url,
        license=_extract(meta, "LicenseShortName")
        or _extract(meta, "UsageTerms")
        or "UNKNOWN",
        attribution=_extract(meta, "Artist") or _extract(meta, "Credit"),
        description=_extract(meta, "ImageDescription"),
        categories=categories,
        width=info.get("width"),
        height=info.get("height"),
        klass=klass,
    )


class WikimediaSource(Source):
    """Commons por categoría o por búsqueda de texto."""

    name = "wikimedia"

    def search(
        self, query: str, klass: ImageClass, limit: int = 200
    ) -> Iterator[ImageRecord]:
        """`query` es una categoría si empieza por `Category:`, si no, texto libre."""
        if query.lower().startswith("category:"):
            yield from self.by_category(query.split(":", 1)[1], klass, limit)
        else:
            yield from self.by_text(query, klass, limit)

    # --- modos de descubrimiento -----------------------------------------

    def by_category(
        self, category: str, klass: ImageClass, limit: int = 200
    ) -> Iterator[ImageRecord]:
        params = {
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": "100",
        }
        yield from self._paginate(params, klass, category, limit)

    def by_text(
        self, text: str, klass: ImageClass, limit: int = 200
    ) -> Iterator[ImageRecord]:
        params = {
            "generator": "search",
            "gsrsearch": text,
            "gsrnamespace": "6",  # namespace File
            "gsrlimit": "50",
        }
        yield from self._paginate(params, klass, text, limit)

    # --- transporte -------------------------------------------------------

    def _paginate(
        self, generator: dict, klass: ImageClass, hint: str, limit: int
    ) -> Iterator[ImageRecord]:
        session = http_session()
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo|categories",
            "iiprop": "url|size|extmetadata",
            "cllimit": "50",
            **generator,
        }

        seen = 0
        while seen < limit:
            try:
                resp = session.get(API, params=params, timeout=45)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"    aviso: wikimedia '{hint}': {str(exc)[:70]}")
                return

            for page in data.get("query", {}).get("pages", []):
                rec = record_from_page(page, klass, hint)
                if rec is None:
                    continue
                yield rec
                seen += 1
                if seen >= limit:
                    return

            cont = data.get("continue")
            if not cont:
                return
            params.update(cont)
