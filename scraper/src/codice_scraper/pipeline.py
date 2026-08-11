"""Orquestación: descubrir, deduplicar, descargar, medir.

El orden importa. Se deduplica **antes** de descargar, contra el manifest y
contra lo ya visto en esta corrida, porque bajar 4 MB para descubrir que la
imagen ya estaba es el desperdicio más caro del proceso. El pHash sólo puede
comprobarse después de bajar, pero la URL y el título no.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .filters import FilterConfig, apply_filters, measure
from .hashing import sha256_file
from .licenses import explain, is_allowed
from .manifest import Manifest
from .models import ImageClass, ImageRecord, UNKNOWN_LICENSE
from .sources.base import Source, http_session


@dataclass
class FetchStats:
    """Resultado de una corrida de descubrimiento y descarga."""

    discovered: int = 0
    skipped_known: int = 0
    skipped_license: int = 0
    downloaded: int = 0
    failed: int = 0
    rejected: int = 0
    duplicates: int = 0
    by_source: Counter = field(default_factory=Counter)
    by_class: Counter = field(default_factory=Counter)
    licenses: Counter = field(default_factory=Counter)
    denied_licenses: Counter = field(default_factory=Counter)
    download_errors: Counter = field(default_factory=Counter)
    dry_run: bool = False

    def render(self) -> str:
        lines = [
            "",
            "=" * 66,
            " FETCH" + ("  (dry-run: no se descargó nada)" if self.dry_run else ""),
            "=" * 66,
            f"  Descubiertas          {self.discovered}",
            f"  Ya conocidas          {self.skipped_known}",
        ]
        if self.skipped_license:
            lines.append(f"  Licencia no apta      {self.skipped_license}")
        if not self.dry_run:
            lines += [
                f"  Descargadas           {self.downloaded}",
                f"  Fallidas              {self.failed}",
                f"  Duplicadas (pHash)    {self.duplicates}",
                f"  Descartadas al filtrar{self.rejected:6d}",
            ]

        if self.by_class:
            lines += ["", "  POR CLASE"]
            for klass, n in self.by_class.most_common():
                lines.append(f"    {klass:30s} {n:4d}")
        if self.by_source:
            lines += ["", "  POR FUENTE"]
            for src, n in self.by_source.most_common():
                lines.append(f"    {src:30s} {n:4d}")
        if self.licenses:
            lines += ["", "  LICENCIAS ADMITIDAS"]
            for lic, n in self.licenses.most_common(10):
                lines.append(f"    {lic[:38]:38s} {n:4d}")
        if self.denied_licenses:
            lines += ["", "  RECHAZADAS POR LICENCIA"]
            for lic, n in self.denied_licenses.most_common(8):
                lines.append(f"    {lic[:52]:52s} {n:4d}")
        if self.download_errors:
            lines += ["", "  ERRORES DE DESCARGA (fuente: tipo de excepción)"]
            for err, n in self.download_errors.most_common(8):
                lines.append(f"    {err[:52]:52s} {n:4d}")

        lines.append("=" * 66)
        return "\n".join(lines)


def _download(
    rec: ImageRecord, dest_dir: Path, sources_by_name: dict[str, Source]
) -> tuple[ImageRecord, str | None]:
    """Descarga una imagen. Devuelve (record, error).

    El error se propaga con el nombre de la excepción y un fragmento del
    mensaje en vez de tragarse: una fuente que falla en masa (como
    `europepmc` antes de que su descarga se corrigiera para pasar por el ZIP
    de figuras) es indistinguible de un problema real de red si el motivo no
    queda registrado en algún sitio.
    """
    path = dest_dir / rec.filename
    if path.exists():
        return rec, None

    source = sources_by_name.get(rec.source)
    try:
        data = source.download(rec) if source is not None else None
        if data is None:
            resp = http_session().get(rec.download_url, timeout=90, stream=True)  # type: ignore[arg-type]
            resp.raise_for_status()
            chunks = bytearray()
            for chunk in resp.iter_content(1 << 16):
                chunks.extend(chunk)
            data = bytes(chunks)

        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(data)
        # Renombrado atómico: una descarga cortada no deja un archivo a medias
        # que la siguiente corrida daría por bueno.
        tmp.replace(path)
        return rec, None
    except Exception as exc:
        path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)
        return rec, f"{type(exc).__name__}: {str(exc)[:80]}"


def fetch(
    sources: list[Source],
    queries: dict[ImageClass, list[str]],
    manifest: Manifest,
    dest_dir: Path,
    limit_per_query: int = 100,
    dry_run: bool = False,
    allow_unknown_license: bool = False,
    cfg: FilterConfig | None = None,
    workers: int = 4,
) -> FetchStats:
    """Descubre con cada fuente, descarga lo nuevo y lo mide."""
    cfg = cfg or FilterConfig()
    stats = FetchStats(dry_run=dry_run)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    known_urls = manifest.known_urls()
    known_phash = manifest.known_phash()
    seen_urls: set[str] = set()
    candidates: list[ImageRecord] = []

    for source in sources:
        if not source.available:
            print(f"  {source.name}: sin credenciales, se omite")
            continue

        for klass, terms in queries.items():
            for term in terms:
                found = 0
                for rec in source.search(term, klass, limit_per_query):
                    stats.discovered += 1

                    if not rec.download_url or rec.download_url in known_urls:
                        stats.skipped_known += 1
                        continue
                    if rec.download_url in seen_urls:
                        stats.skipped_known += 1
                        continue
                    # Política de licencias única para todas las fuentes.
                    # Aplicarla aquí y no dentro de cada `Source` evita que
                    # diverjan (y que reaparezca el fallo de dejar pasar NC/ND
                    # por coincidencia de subcadena).
                    if not is_allowed(rec.license):
                        unknown = rec.license == UNKNOWN_LICENSE
                        if not (unknown and allow_unknown_license):
                            stats.skipped_license += 1
                            stats.denied_licenses[
                                f"{rec.license[:34]} — {explain(rec.license)}"
                            ] += 1
                            continue

                    seen_urls.add(rec.download_url)
                    candidates.append(rec)
                    stats.by_source[source.name] += 1
                    stats.by_class[klass.value] += 1
                    stats.licenses[rec.license] += 1
                    found += 1

                print(f"  {source.name:12s} {klass.value:26s} {term[:28]:28s} -> {found}")

    if dry_run or not candidates:
        return stats

    sources_by_name = {s.name: s for s in sources}

    print(f"\nDescargando {len(candidates)} imágenes…")
    results: list[ImageRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download, rec, dest_dir, sources_by_name) for rec in candidates
        ]
        for fut in tqdm(futures, desc="  bajando", unit="img"):
            rec, err = fut.result()
            if err:
                stats.failed += 1
                stats.download_errors[f"{rec.source}: {err.split(':')[0]}"] += 1
                continue
            stats.downloaded += 1
            results.append(rec)

    print("\nMidiendo…")
    for rec in tqdm(results, desc="  midiendo", unit="img"):
        path = dest_dir / rec.filename
        try:
            metrics = measure(path, cfg)
        except Exception:
            path.unlink(missing_ok=True)
            stats.failed += 1
            continue

        rec.sha256 = sha256_file(path)
        apply_filters(rec, metrics, cfg)

        # El pHash sólo se puede comprobar tras descargar: distintas fuentes
        # sirven la misma foto bajo URLs distintas.
        if rec.phash and rec.phash in known_phash:
            from .models import RejectReason

            rec.reject(RejectReason.DUPLICATE)
            stats.duplicates += 1
        elif rec.phash:
            known_phash[rec.phash] = rec.filename

        if rec.rejected:
            stats.rejected += 1
        manifest.upsert(rec)

    return stats
