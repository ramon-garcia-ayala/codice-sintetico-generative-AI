"""Flickr, restringido a licencias Creative Commons.

Aporta lo que las fuentes institucionales no tienen: fotografía de campo y
macro hecha por gente que estaba ahí, con encuadres menos catalográficos. Útil
sobre todo para las clases escasas — plastiglomerado y proxies materiales.

Necesita una API key gratuita (https://www.flickr.com/services/apps/create/).
Va en `config/secrets.yaml`, que está en `.gitignore`:

    flickr:
      api_key: "..."

Sin key la fuente queda deshabilitada en vez de fallar a mitad de una corrida.
"""

from __future__ import annotations

from collections.abc import Iterator

from ..models import ImageClass, ImageRecord
from .base import Source, http_session

API = "https://www.flickr.com/services/rest/"

#: IDs de licencia de Flickr que permiten reutilización.
#: 1 BY-NC-SA · 2 BY-NC · 3 BY-NC-ND · 4 BY · 5 BY-SA · 6 BY-ND
#: 7 dominio público · 9 CC0 · 10 dominio público (marca)
#: Se excluyen las NC por lo mismo que en Europe PMC: el proyecto se expone en
#: una institución y puede tener difusión comercial.
CC_LICENSES = "4,5,7,9,10"

_LICENSE_NAMES = {
    "4": "CC BY 2.0",
    "5": "CC BY-SA 2.0",
    "7": "Dominio público (Flickr Commons)",
    "9": "CC0",
    "10": "Dominio público",
}

#: Tamaños de mayor a menor: `url_o` (original) no siempre está disponible.
_SIZE_KEYS = ("url_o", "url_k", "url_h", "url_b")


class FlickrSource(Source):
    """Búsqueda en Flickr filtrada por licencia CC."""

    name = "flickr"
    requires_key = True

    def search(
        self, query: str, klass: ImageClass, limit: int = 100
    ) -> Iterator[ImageRecord]:
        if not self.available:
            print("    aviso: flickr sin api_key, se omite (ver config/secrets.yaml)")
            return

        session = http_session()
        page = 1
        yielded = 0

        while yielded < limit:
            try:
                resp = session.get(
                    API,
                    params={
                        "method": "flickr.photos.search",
                        "api_key": self.config["api_key"],
                        "text": query,
                        "license": CC_LICENSES,
                        "content_type": "1",  # sólo fotos
                        "media": "photos",
                        "sort": "relevance",
                        "safe_search": "1",
                        "per_page": "100",
                        "page": str(page),
                        "extras": "license,owner_name,description,"
                        + ",".join(_SIZE_KEYS)
                        + ",o_dims",
                        "format": "json",
                        "nojsoncallback": "1",
                    },
                    timeout=45,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"    aviso: flickr '{query}': {str(exc)[:70]}")
                return

            if data.get("stat") != "ok":
                print(f"    aviso: flickr respondió {data.get('message', 'error')}")
                return

            photos = data.get("photos", {})
            items = photos.get("photo", [])
            if not items:
                return

            for photo in items:
                rec = self._to_record(photo, klass, query)
                if rec is None:
                    continue
                yield rec
                yielded += 1
                if yielded >= limit:
                    return

            if page >= int(photos.get("pages", 1)):
                return
            page += 1

    @staticmethod
    def _to_record(photo: dict, klass: ImageClass, query: str) -> ImageRecord | None:
        url = width = height = None
        for key in _SIZE_KEYS:
            if photo.get(key):
                url = photo[key]
                width = int(photo.get(key.replace("url_", "width_"), 0) or 0)
                height = int(photo.get(key.replace("url_", "height_"), 0) or 0)
                break

        if not url:
            return None
        if width and height and min(width, height) < 1024:
            return None

        photo_id = photo.get("id", "")
        owner = photo.get("ownername") or photo.get("owner", "")
        license_id = str(photo.get("license", ""))

        return ImageRecord(
            filename=f"flickr_{photo_id}.jpg",
            source="flickr",
            source_id=photo_id,
            origin_title=photo.get("title") or f"Flickr {photo_id}",
            origin_url=f"https://www.flickr.com/photos/{photo.get('owner', '')}/{photo_id}",
            download_url=url,
            license=_LICENSE_NAMES.get(license_id, f"Flickr license {license_id}"),
            attribution=owner,
            description=(photo.get("description") or {}).get("_content"),
            categories=[query, "flickr"],
            width=width or None,
            height=height or None,
            klass=klass,
        )
