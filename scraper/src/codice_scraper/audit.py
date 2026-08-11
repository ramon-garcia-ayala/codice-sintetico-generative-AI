"""Auditoría de un directorio de imágenes.

Responde a la pregunta con la que arrancó este paquete: qué hay realmente en
`02_DATASET` y cuánto sirve. Es reejecutable y su salida es el test de
regresión de los umbrales de `filters`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from .filters import FilterConfig, ImageMetrics, apply_filters, iter_images, measure
from .models import ImageRecord, RejectReason


@dataclass
class AuditReport:
    """Resumen agregado de un directorio."""

    root: Path
    total_files: int = 0
    readable: int = 0
    unreadable: list[tuple[str, str]] = field(default_factory=list)

    short_sides: list[int] = field(default_factory=list)
    aspects: list[float] = field(default_factory=list)
    saturations: list[float] = field(default_factory=list)
    sharpness: list[float] = field(default_factory=list)
    total_bytes: int = 0

    modes: dict[str, int] = field(default_factory=dict)
    suffixes: dict[str, int] = field(default_factory=dict)

    studio: list[str] = field(default_factory=list)
    too_small: list[str] = field(default_factory=list)
    blurry: list[str] = field(default_factory=list)
    panoramic: list[str] = field(default_factory=list)
    extreme_aspect: list[str] = field(default_factory=list)

    has_captions: int = 0
    duplicates: dict[str, str] = field(default_factory=dict)

    @property
    def usable(self) -> int:
        bad = set(self.studio) | set(self.too_small) | set(self.blurry)
        bad |= set(self.extreme_aspect) | set(self.duplicates)
        return self.readable - len(bad)

    def render(self) -> str:
        """Reporte legible en terminal."""
        if not self.readable:
            return f"Sin imágenes legibles en {self.root}"

        def pct(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            return s[min(int(len(s) * p), len(s) - 1)]

        n = self.readable
        share = lambda k: f"{100 * k // n}%"  # noqa: E731

        lines = [
            "",
            "=" * 66,
            f" AUDITORÍA  {self.root}",
            "=" * 66,
            f"  Archivos totales      {self.total_files}",
            f"  Imágenes legibles     {self.readable}",
        ]
        if self.unreadable:
            lines.append(f"  Ilegibles             {len(self.unreadable)}")
        lines += [
            f"  Captions (.txt)       {self.has_captions}"
            + ("   <-- BLOQUEANTE para kohya" if self.has_captions < n else ""),
            f"  Peso total            {self.total_bytes / 1e6:,.0f} MB"
            f"  ({self.total_bytes / n / 1e6:.2f} MB/img)",
            "",
            "  RESOLUCIÓN (lado corto)",
            f"    min {min(self.short_sides)}   p10 {pct(self.short_sides, .10):.0f}"
            f"   mediana {statistics.median(self.short_sides):.0f}"
            f"   p90 {pct(self.short_sides, .90):.0f}   max {max(self.short_sides)}",
            "",
            "  ASPECTO",
            f"    mediana {statistics.median(self.aspects):.2f}"
            f"   p90 {pct(self.aspects, .90):.2f}   max {max(self.aspects):.2f}",
            "",
            "  NITIDEZ (varianza del laplaciano)",
            f"    min {min(self.sharpness):.0f}   p10 {pct(self.sharpness, .10):.0f}"
            f"   mediana {statistics.median(self.sharpness):.0f}"
            f"   max {max(self.sharpness):.0f}",
            "",
            "  VEREDICTOS",
            f"    Fondo de estudio / vitrina   {len(self.studio):4d}  {share(len(self.studio))}",
            f"    Bajo el mínimo de resolución {len(self.too_small):4d}  {share(len(self.too_small))}",
            f"    Desenfocadas                 {len(self.blurry):4d}  {share(len(self.blurry))}",
            f"    Aspecto extremo              {len(self.extreme_aspect):4d}",
            f"    Duplicadas                   {len(self.duplicates):4d}",
            f"    Panorámicas (bucket aparte)  {len(self.panoramic):4d}",
            "",
            f"    APROVECHABLES                {self.usable:4d}  {share(self.usable)}",
            "",
            "  FORMATOS",
            f"    {dict(sorted(self.suffixes.items()))}",
            f"    modos: {dict(sorted(self.modes.items()))}",
        ]

        odd_modes = {k: v for k, v in self.modes.items() if k not in ("RGB", "L")}
        if odd_modes:
            lines.append(f"    a normalizar a RGB: {odd_modes}")

        if self.studio:
            lines += ["", "  Muestra de fondo de estudio (primeras 12):"]
            lines += [f"    {n}" for n in self.studio[:12]]

        lines.append("=" * 66)
        return "\n".join(lines)


def audit_directory(
    root: Path,
    cfg: FilterConfig | None = None,
    records: dict[str, ImageRecord] | None = None,
    show_progress: bool = True,
) -> tuple[AuditReport, list[ImageRecord]]:
    """Mide todas las imágenes de `root` y devuelve reporte y records.

    Si se pasa `records`, se reutilizan los que ya tengan métricas (por
    ejemplo del manifest) en vez de volver a medirlos.
    """
    cfg = cfg or FilterConfig()
    root = Path(root)
    report = AuditReport(root=root)

    all_files = [p for p in root.iterdir() if p.is_file()]
    report.total_files = len(all_files)

    images = iter_images(root)
    out: list[ImageRecord] = []

    iterator = tqdm(images, desc="  midiendo", unit="img") if show_progress else images
    for path in iterator:
        rec = (records or {}).get(path.name) or ImageRecord(
            filename=path.name, source="local"
        )
        try:
            m: ImageMetrics = measure(path, cfg)
        except Exception as exc:
            report.unreadable.append((path.name, str(exc)[:80]))
            rec.reject(RejectReason.UNREADABLE)
            out.append(rec)
            continue

        apply_filters(rec, m, cfg)
        out.append(rec)

        report.readable += 1
        report.short_sides.append(m.short_side)
        report.aspects.append(m.aspect)
        report.saturations.append(m.saturation)
        report.sharpness.append(m.sharpness)
        report.total_bytes += m.bytes
        report.modes[m.mode] = report.modes.get(m.mode, 0) + 1
        suffix = path.suffix.lower()
        report.suffixes[suffix] = report.suffixes.get(suffix, 0) + 1

        if path.with_suffix(".txt").exists():
            report.has_captions += 1

        if RejectReason.STUDIO_BACKGROUND in rec.reject_reasons:
            report.studio.append(path.name)
        if RejectReason.TOO_SMALL in rec.reject_reasons:
            report.too_small.append(path.name)
        if RejectReason.BLURRY in rec.reject_reasons:
            report.blurry.append(path.name)
        if RejectReason.EXTREME_ASPECT in rec.reject_reasons:
            report.extreme_aspect.append(path.name)
        if rec.panoramic:
            report.panoramic.append(path.name)

    from .filters import find_duplicates

    report.duplicates = find_duplicates(out, cfg)
    for name in report.duplicates:
        rec = next((r for r in out if r.filename == name), None)
        if rec:
            rec.reject(RejectReason.DUPLICATE)

    return report, out
