"""Rescate de procedencia por pHash contra Wikimedia Commons.

El scrape original de `02_DATASET` renombró todo a `wikimedia_###` sin guardar
manifest, lo que borró título, autor y licencia de las 282 imágenes. Sin eso no
se puede ni construir captions desde metadatos ni atribuir el material.

La reconstrucción es posible porque las imágenes siguen existiendo en Commons:
se barren las mismas categorías, se calcula el pHash de cada miniatura remota y
se empareja contra los pHash locales. Un match devuelve la ficha completa.

El pHash es robusto al reescalado y a la recompresión, que es exactamente la
diferencia entre el archivo local y la miniatura que sirve Commons.
"""

from __future__ import annotations

import io
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import requests
import yaml
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

from .hashing import hamming, phash
from .licenses import is_allowed
from .manifest import Manifest
from .models import ImageRecord, RejectReason

COMMONS_API = "https://commons.wikimedia.org/w/api.php"

#: Commons exige un User-Agent identificable con forma de contacto.
USER_AGENT = (
    "codice-sintetico-dataset/0.1 "
    "(https://github.com/codice-sintetico; proyecto Fondo Creativo 2026) "
    "python-requests"
)

#: Ancho de la miniatura que pedimos. Suficiente para un pHash estable y unas
#: 40 veces más ligera que el original.
THUMB_WIDTH = 320

#: Umbral de match. 5 es el mismo que usa la deduplicación; por encima de ~12
#: los falsos positivos se disparan en imágenes de roca, que son texturalmente
#: parecidas entre sí.
MATCH_THRESHOLD = 5

#: Umbrales extra que sólo se reportan, para poder recalibrar con datos.
DIAGNOSTIC_THRESHOLDS = (5, 8, 12, 16)


@dataclass
class RecoveryStats:
    """Resultado de una corrida de rescate."""

    local_total: int = 0
    candidates_seen: int = 0
    thumbs_ok: int = 0
    thumbs_failed: int = 0
    matched: int = 0
    already_had_provenance: int = 0
    licenses: Counter = field(default_factory=Counter)
    distance_buckets: Counter = field(default_factory=Counter)
    errors: Counter = field(default_factory=Counter)
    unmatched: list[str] = field(default_factory=list)
    dry_run: bool = False

    def render(self) -> str:
        pct = (100 * self.matched // self.local_total) if self.local_total else 0
        lines = [
            "",
            "=" * 66,
            " RESCATE DE PROCEDENCIA  (pHash vs Wikimedia Commons)",
            "=" * 66,
            f"  Imagenes locales sin procedencia   {self.local_total}",
            f"  Candidatas consultadas en Commons  {self.candidates_seen}",
            f"  Miniaturas descargadas             {self.thumbs_ok}"
            + (f"   (fallidas {self.thumbs_failed})" if self.thumbs_failed else ""),
            "",
            f"  EMPAREJADAS                        {self.matched}/{self.local_total}  ({pct}%)",
            f"  Sin emparejar                      {len(self.unmatched)}",
        ]
        if self.already_had_provenance:
            lines.append(f"  Ya tenian procedencia              {self.already_had_provenance}")

        if self.distance_buckets:
            lines += ["", "  DISTANCIAS DE MATCH (para recalibrar el umbral)"]
            for t in DIAGNOSTIC_THRESHOLDS:
                n = sum(v for k, v in self.distance_buckets.items() if k <= t)
                mark = "  <-- umbral activo" if t == MATCH_THRESHOLD else ""
                lines.append(f"    Hamming <= {t:2d}   {n:4d}{mark}")

        if self.errors:
            lines += ["", "  ERRORES DE DESCARGA"]
            for kind, n in self.errors.most_common(6):
                lines.append(f"    {kind:40s} {n:4d}")

        if self.licenses:
            lines += ["", "  LICENCIAS RECUPERADAS"]
            for lic, n in self.licenses.most_common():
                lines.append(f"    {lic:40s} {n:4d}")

        if self.unmatched:
            lines += ["", f"  Sin emparejar (primeras 15 de {len(self.unmatched)}):"]
            lines += [f"    {n}" for n in self.unmatched[:15]]

        lines.append("=" * 66)
        return "\n".join(lines)


#: Peticiones concurrentes contra upload.wikimedia.org. Conservador a
#: propósito: con 8 hilos Commons empieza a cortar ráfagas y la tasa de fallo
#: se dispara. 4 con reintentos va más rápido en la práctica que 8 sin ellos.
DEFAULT_WORKERS = 4

_local = threading.local()


def _session() -> requests.Session:
    """Sesión HTTP con reintentos y pool dimensionado.

    `requests.Session` no es segura de compartir entre hilos, así que se
    guarda una por hilo en almacenamiento thread-local.
    """
    session = getattr(_local, "session", None)
    if session is not None:
        return session

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "image/*,*/*"})
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    _local.session = session
    return session


def load_categories(config_path: Path | None = None) -> list[str]:
    """Todas las categorías de `queries.yaml`, en un solo listado plano."""
    config_path = config_path or (
        Path(__file__).resolve().parents[2] / "config" / "queries.yaml"
    )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cats: list[str] = []
    for group in (data.get("wikimedia_categories") or {}).values():
        cats.extend(group or [])
    # dedup preservando orden
    return list(dict.fromkeys(cats))


def iter_category_files(
    session: requests.Session, category: str, limit: int = 600
) -> list[dict]:
    """Archivos de una categoría de Commons, con metadatos y URL de miniatura.

    Un solo `generator=categorymembers` con `prop=imageinfo` trae la ficha
    completa, así que no hace falta una segunda ronda de peticiones por archivo.
    """
    out: list[dict] = []
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "generator": "categorymembers",
        "gcmtitle": f"Category:{category}",
        "gcmtype": "file",
        "gcmlimit": "100",
        "prop": "imageinfo",
        "iiprop": "url|size|extmetadata",
        "iiurlwidth": str(THUMB_WIDTH),
    }

    while len(out) < limit:
        try:
            resp = session.get(COMMONS_API, params=params, timeout=45)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"    aviso: fallo consultando '{category}': {str(exc)[:70]}")
            break

        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("thumburl"):
                out.append({"title": page.get("title", ""), "info": info})

        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)

    return out[:limit]


def _extract(meta: dict, key: str) -> str | None:
    """Saca un campo de `extmetadata` y le quita el HTML que Commons incrusta."""
    import re

    value = (meta.get(key) or {}).get("value")
    if not value:
        return None
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _fetch_thumb_phash(cand: dict) -> tuple[dict | None, str | None, str | None]:
    """Descarga la miniatura y calcula su pHash.

    Devuelve `(candidata, phash, None)` o `(None, None, error)`. El error se
    propaga en vez de tragarse: una tanda que falla en masa suele ser rate
    limiting o un header mal puesto, y sin el mensaje no hay forma de saberlo.
    """
    url = cand["info"]["thumburl"]
    try:
        resp = _session().get(url, timeout=45)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        return cand, phash(img.convert("RGB")), None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {str(exc)[:90]}"


def recover_provenance(
    manifest: Manifest,
    root: Path,
    dry_run: bool = False,
    limit_per_category: int = 600,
    categories: list[str] | None = None,
    threshold: int = MATCH_THRESHOLD,
    workers: int = DEFAULT_WORKERS,
) -> RecoveryStats:
    """Empareja las imágenes locales con Commons y rellena su procedencia."""
    stats = RecoveryStats(dry_run=dry_run)
    session = _session()

    pending: dict[str, ImageRecord] = {}
    for rec in manifest.all():
        if rec.has_provenance:
            stats.already_had_provenance += 1
            continue
        if rec.phash:
            pending[rec.filename] = rec

    stats.local_total = len(pending)
    if not pending:
        print("Nada que rescatar: o no hay pHash calculado (corre `audit "
              "--write-manifest` primero) o ya todo tiene procedencia.")
        return stats

    cats = categories or load_categories()
    print(f"{len(pending)} imágenes sin procedencia. Barriendo {len(cats)} categorías…\n")

    for category in cats:
        if not pending:
            break

        candidates = iter_category_files(session, category, limit_per_category)
        stats.candidates_seen += len(candidates)
        if not candidates:
            continue

        desc = f"  {category[:34]:34s}"
        results: list[tuple[dict, str]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch_thumb_phash, c) for c in candidates]
            for fut in tqdm(futures, desc=desc, unit="img", leave=False):
                cand, remote_hash, err = fut.result()
                if err:
                    stats.thumbs_failed += 1
                    stats.errors[err.split(":")[0]] += 1
                else:
                    results.append((cand, remote_hash))  # type: ignore[arg-type]
                    stats.thumbs_ok += 1

        matched_now = 0
        for cand, remote_hash in results:
            if not pending:
                break
            best_name, best_dist = None, 999
            for name, rec in pending.items():
                d = hamming(remote_hash, rec.phash)  # type: ignore[arg-type]
                if d < best_dist:
                    best_name, best_dist = name, d
                    if d == 0:
                        break

            stats.distance_buckets[best_dist] += 1
            if best_name is None or best_dist > threshold:
                continue

            rec = pending.pop(best_name)
            _apply_metadata(rec, cand, category)
            stats.licenses[rec.license] += 1
            stats.matched += 1
            matched_now += 1
            if not dry_run:
                manifest.upsert(rec)

        print(f"  {category[:34]:34s} {len(candidates):4d} candidatas -> "
              f"{matched_now:3d} emparejadas  ({len(pending)} pendientes)")

    stats.unmatched = sorted(pending)
    return stats


def _apply_metadata(rec: ImageRecord, cand: dict, category: str) -> None:
    """Vuelca la ficha de Commons sobre el record local y aplica la licencia.

    El gate de licencias tiene que estar aquí y no sólo en `pipeline.fetch`:
    las imágenes heredadas del Drive nunca pasan por `fetch` —entran con
    `ingest` y reciben su licencia justo en esta función—, así que sin esta
    comprobación el camino ingest → recover → export se salta la política
    entera y publica en `ATTRIBUTIONS.md` como verificado algo que `fetch`
    habría bloqueado.
    """
    info = cand["info"]
    meta = info.get("extmetadata") or {}

    rec.source = "wikimedia"
    rec.origin_title = cand["title"]
    rec.origin_url = info.get("descriptionurl")
    rec.download_url = info.get("url")
    rec.license = (
        _extract(meta, "LicenseShortName") or _extract(meta, "UsageTerms") or rec.license
    )
    rec.attribution = _extract(meta, "Artist") or _extract(meta, "Credit")
    rec.description = _extract(meta, "ImageDescription")

    cats = _extract(meta, "Categories")
    rec.categories = [c.strip() for c in cats.split("|") if c.strip()] if cats else [category]
    if category not in rec.categories:
        rec.categories.append(category)

    apply_license_policy(rec)


def apply_license_policy(rec: ImageRecord) -> bool:
    """Marca o levanta `LICENSE_DENIED` según la licencia actual del record.

    Idempotente y reversible: si una corrida posterior recupera una licencia
    mejor para la misma imagen, el descarte se retira solo. Devuelve True si la
    licencia es admisible.
    """
    allowed = is_allowed(rec.license)
    has_flag = RejectReason.LICENSE_DENIED in rec.reject_reasons

    if allowed and has_flag:
        rec.reject_reasons.remove(RejectReason.LICENSE_DENIED)
        rec.rejected = bool(rec.reject_reasons)
    elif not allowed and not has_flag:
        rec.reject(RejectReason.LICENSE_DENIED)

    return allowed
