"""Manifest del dataset: un JSONL con la procedencia de cada imagen.

Es la única fuente de verdad sobre de dónde salió cada archivo y bajo qué
licencia. El scrape original de `02_DATASET` no tenía manifest y por eso se
perdió irreversiblemente la procedencia de las 282 imágenes al renombrarlas a
`wikimedia_###`; este módulo existe para que eso no vuelva a pasar.

Formato JSONL, una línea por imagen, indexado por `filename`. Se reescribe
completo al guardar (compactación) en vez de anexar ciegamente, para que
reejecutar el scraper sea idempotente.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import ImageRecord


class Manifest:
    """Colección de `ImageRecord` persistida en JSONL."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: dict[str, ImageRecord] = {}

    # --- persistencia -----------------------------------------------------

    def load(self) -> Manifest:
        """Carga el JSONL si existe. Las líneas corruptas se ignoran con aviso."""
        self._records.clear()
        if not self.path.exists():
            return self

        with open(self.path, encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = ImageRecord.model_validate_json(line)
                except Exception as exc:  # línea corrupta: no tumbar el manifest entero
                    print(f"  aviso: {self.path.name}:{lineno} ilegible ({exc})")
                    continue
                self._records[rec.filename] = rec
        return self

    def save(self) -> None:
        """Reescribe el JSONL completo, ordenado por filename."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for name in sorted(self._records):
                fh.write(self._records[name].model_dump_json(exclude_none=True) + "\n")
        tmp.replace(self.path)

    # --- acceso -----------------------------------------------------------

    def upsert(self, rec: ImageRecord) -> None:
        """Inserta o reemplaza por `filename`. Es lo que hace idempotente al scraper."""
        self._records[rec.filename] = rec

    def get(self, filename: str) -> ImageRecord | None:
        return self._records.get(filename)

    def all(self) -> list[ImageRecord]:
        return [self._records[k] for k in sorted(self._records)]

    def active(self) -> list[ImageRecord]:
        """Los que sobreviven al filtrado."""
        return [r for r in self.all() if not r.rejected]

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, filename: str) -> bool:
        return filename in self._records

    # --- consultas de deduplicación ---------------------------------------

    def known_sha256(self) -> set[str]:
        return {r.sha256 for r in self._records.values() if r.sha256}

    def known_phash(self) -> dict[str, str]:
        """pHash -> filename, para detectar recapturas de lo ya descargado."""
        return {r.phash: r.filename for r in self._records.values() if r.phash}

    def known_urls(self) -> set[str]:
        """URLs ya vistas: permite saltarse descargas sin tocar la red."""
        return {r.download_url for r in self._records.values() if r.download_url}


def write_attributions(manifest: Manifest, path: Path) -> int:
    """Genera ATTRIBUTIONS.md a partir del manifest. Devuelve cuántas entradas.

    Sirve tanto para el fanzine como para responder a la institución sobre la
    procedencia del material de entrenamiento.
    """
    from .models import UNKNOWN_LICENSE

    records = [r for r in manifest.active() if r.origin_title]
    unknown = [r for r in manifest.active() if r.license == UNKNOWN_LICENSE]

    lines = [
        "# Atribuciones del dataset",
        "",
        "Generado automáticamente por `codice-scraper report --attributions`.",
        "No editar a mano: los cambios se pierden en la siguiente corrida.",
        "",
        f"- Imágenes activas: **{len(manifest.active())}**",
        f"- Con procedencia identificada: **{len(records)}**",
        f"- Sin licencia identificada: **{len(unknown)}**",
        "",
    ]

    by_source: dict[str, list] = {}
    for rec in records:
        by_source.setdefault(rec.source, []).append(rec)

    for source in sorted(by_source):
        lines += [f"## {source}", ""]
        for rec in sorted(by_source[source], key=lambda r: r.filename):
            title = rec.origin_title or rec.filename
            parts = [f"- **{rec.filename}** — {title}"]
            if rec.attribution:
                parts.append(f"por {rec.attribution}")
            parts.append(f"({rec.license})")
            if rec.origin_url:
                parts.append(f"<{rec.origin_url}>")
            lines.append(" ".join(parts))
        lines.append("")

    if unknown:
        lines += [
            "## Sin procedencia identificada",
            "",
            "Estas imágenes no pudieron emparejarse con su fuente original.",
            "Decidir caso por caso si entran al entrenamiento.",
            "",
        ]
        lines += [f"- {r.filename}" for r in sorted(unknown, key=lambda r: r.filename)]
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(records)
