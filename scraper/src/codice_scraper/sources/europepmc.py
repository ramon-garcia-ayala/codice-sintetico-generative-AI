"""Europe PMC — figuras de artículos de acceso abierto.

Es la fuente que de verdad importa para el plastiglomerado. El término se acuñó
en 2014 (Corcoran, Jazvac & Moore) y casi todo el material fotográfico
existente vive dentro de literatura científica, no en bancos de imágenes.
Wikimedia Commons tiene del orden de decenas de fotos; los papers de
plastiglomerado y *pyroplastics* tienen bastantes más.

Dos cautelas propias de esta fuente:

- Sólo se aceptan artículos con licencia CC, que Europe PMC declara por
  artículo. Acceso abierto no implica reutilización libre.
- Las figuras científicas suelen ser compuestas (paneles a, b, c con flechas y
  barras de escala). **No se recortan automáticamente**: se marcan en
  `needs_review` para que el recorte sea una decisión con la imagen delante.
"""

from __future__ import annotations

import io
import re
import threading
import zipfile
from collections.abc import Iterator

from ..licenses import is_allowed
from ..models import ImageClass, ImageRecord
from .base import Source, http_session

SEARCH_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

#: `europepmc.org/articles/{pmcid}/bin/{name}` (el enlace "natural" que se ve
#: en la página del artículo) redirige a un backend CGI que corta la conexión
#: ante cualquier cliente no-navegador — probado con varios User-Agent y con
#: Referer, todos con el mismo `RemoteDisconnected`. El mismo PMCID en NCBI
#: directo devuelve 403 explícito. Ninguno de los dos sirve para scraping.
#:
#: La vía que sí funciona, verificada contra 3 artículos: el paquete de
#: "supplementary files" que expone la propia API REST de EBI (dominio
#: `ebi.ac.uk`, no `europepmc.org`) — un ZIP con todas las figuras del
#: artículo, JPG y GIF. Se descarga una vez por artículo, no una vez por
#: figura, y se cachea en `_zip_cache` para no repetirlo.
SUPPLEMENTARY_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles"

#: Separador entre la URL del ZIP y el nombre del miembro dentro de él, usado
#: para codificar ambos datos en el único campo `download_url` que el resto
#: del pipeline conoce. `::` no aparece en URLs ni en nombres de archivo reales.
_ZIP_MEMBER_SEP = "::"

_zip_cache: dict[str, bytes] = {}
_zip_cache_lock = threading.Lock()


class EuropePMCSource(Source):
    """Figuras de artículos open access con licencia CC."""

    name = "europepmc"

    def download(self, rec: ImageRecord) -> bytes:
        """Extrae la figura del ZIP de `supplementaryFiles` del artículo."""
        zip_url, _, member = (rec.download_url or "").rpartition(_ZIP_MEMBER_SEP)
        if not zip_url:
            raise ValueError(f"download_url mal formado: {rec.download_url!r}")

        data = self._fetch_zip(zip_url)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            chosen = _best_zip_member(member, names)
            if chosen is None:
                raise FileNotFoundError(f"'{member}' no está en {zip_url}")
            return zf.read(chosen)

    @staticmethod
    def _fetch_zip(zip_url: str) -> bytes:
        with _zip_cache_lock:
            cached = _zip_cache.get(zip_url)
        if cached is not None:
            return cached

        resp = http_session().get(zip_url, timeout=60)
        resp.raise_for_status()
        data = resp.content

        with _zip_cache_lock:
            _zip_cache[zip_url] = data
        return data

    def search(
        self, query: str, klass: ImageClass, limit: int = 100
    ) -> Iterator[ImageRecord]:
        session = http_session()
        yielded = 0
        cursor = "*"

        while yielded < limit:
            try:
                resp = session.get(
                    SEARCH_API,
                    params={
                        "query": f'({query}) AND (OPEN_ACCESS:"Y") AND (HAS_FT:"Y")',
                        "format": "json",
                        "pageSize": "50",
                        "cursorMark": cursor,
                        "resultType": "core",
                    },
                    timeout=45,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                print(f"    aviso: europepmc '{query}': {str(exc)[:70]}")
                return

            results = data.get("resultList", {}).get("result", [])
            if not results:
                return

            for article in results:
                pmcid = article.get("pmcid")
                # Filtro temprano con la política centralizada: evita bajar el
                # fullTextXML de artículos cuya licencia ya sabemos que se va
                # a rechazar en el pipeline. `is_allowed` es la misma función
                # que aplica el resto de fuentes — no una copia local que
                # pueda divergir.
                if not pmcid or not is_allowed(article.get("license")):
                    continue

                for rec in self._figures(session, pmcid, article, klass, query):
                    yield rec
                    yielded += 1
                    if yielded >= limit:
                        return

            next_cursor = data.get("nextCursorMark")
            if not next_cursor or next_cursor == cursor:
                return
            cursor = next_cursor

    def _figures(
        self,
        session,
        pmcid: str,
        article: dict,
        klass: ImageClass,
        query: str,
    ) -> Iterator[ImageRecord]:
        """Figuras del texto completo en XML de un artículo."""
        try:
            resp = session.get(
                f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML",
                timeout=45,
            )
            resp.raise_for_status()
            xml = resp.text
        except Exception:
            return

        from xml.etree import ElementTree

        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError:
            return

        title = (article.get("title") or "").strip()
        authors = (article.get("authorString") or "").strip()
        journal = (article.get("journalTitle") or "").strip()

        for fig in root.iter("fig"):
            graphic = next(
                (g for g in fig.iter() if g.tag.endswith("graphic")), None
            )
            if graphic is None:
                continue

            href = next(
                (v for k, v in graphic.attrib.items() if k.endswith("href")), None
            )
            if not href:
                continue

            caption_el = fig.find("caption")
            caption = (
                " ".join("".join(caption_el.itertext()).split())
                if caption_el is not None
                else ""
            )

            # Bug corregido: `r'[^\\w.-]+'` (doble backslash) trata "w" como
            # carácter literal a excluir en vez de la clase \w, así que casi
            # todo el nombre se borraba y figuras distintas colisionaban en el
            # mismo filename. Un solo backslash es la clase de carácter real.
            safe_href = re.sub(r"[^\w.-]+", "_", href)
            if not safe_href.lower().endswith((".jpg", ".jpeg", ".png")):
                safe_href += ".jpg"
            zip_url = SUPPLEMENTARY_URL.format(pmcid=pmcid)

            rec = ImageRecord(
                filename=f"epmc_{pmcid.lower()}_{safe_href}",
                source="europepmc",
                source_id=pmcid,
                origin_title=f"{title} — {fig.get('id', 'fig')}",
                origin_url=f"https://europepmc.org/article/PMC/{pmcid}",
                download_url=f"{zip_url}{_ZIP_MEMBER_SEP}{href}",
                license=article.get("license", "cc by"),
                attribution=f"{authors} ({journal})" if authors else journal,
                description=caption[:500],
                categories=[query, "europepmc"],
                klass=klass,
            )
            # Casi todas las figuras de paper son compuestas. Recortarlas a
            # ciegas produciría paneles partidos por la mitad.
            rec.needs_review.append("figura de paper: revisar si es compuesta y recortar")
            yield rec


def _best_zip_member(wanted: str, names: list[str]) -> str | None:
    """Elige el miembro del ZIP que corresponde a `wanted`.

    El ZIP trae cada figura en dos formatos (`.jpg` de resolución completa y
    un `.gif` mucho más pequeño, probablemente una miniatura). Coincidencia
    exacta primero; si no, mismo nombre base sin importar extensión,
    prefiriendo siempre `.jpg`/`.jpeg`/`.png` sobre `.gif`.
    """
    if wanted in names:
        return wanted

    stem = wanted.rsplit(".", 1)[0].lower()
    same_stem = [n for n in names if n.rsplit(".", 1)[0].lower() == stem]
    if not same_stem:
        return None

    for n in same_stem:
        if n.lower().endswith((".jpg", ".jpeg", ".png")):
            return n
    return same_stem[0]
