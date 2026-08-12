"""Export al formato de entrenamiento de kohya_ss.

kohya espera un directorio por concepto llamado `N_nombre`, donde `N` es el
número de repeticiones por época. Ese prefijo es el mecanismo con el que
balanceamos las cuatro clases sin duplicar archivos en disco: el plastiglomerado
real son unas decenas de imágenes frente a cientos de estratos, y sin repeats
desaparecería dentro del promedio.

Se emite además `captions_SDXL.csv` con cabeceras `file_name,text`, que es lo
que valida `02_SDXL_LoRA_Captions_Check.ipynb` en el repo. Sirve para revisar
el dataset con el flujo de notebooks ya conocido antes de lanzar kohya.
"""

from __future__ import annotations

import csv
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ..licenses import is_allowed
from ..models import CLASS_REPEATS, TRIGGER, ImageClass, ImageRecord


@dataclass
class ExportStats:
    """Resultado de un export."""

    root: Path
    per_class: dict[str, int] = field(default_factory=dict)
    converted: int = 0
    skipped_no_caption: list[str] = field(default_factory=list)
    skipped_license: list[str] = field(default_factory=list)
    missing_source: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.per_class.values())

    @property
    def effective_steps(self) -> int:
        """Imágenes vistas por época, ya multiplicadas por sus repeats."""
        total = 0
        for name, count in self.per_class.items():
            repeats = int(name.split("_", 1)[0])
            total += repeats * count
        return total

    def render(self) -> str:
        lines = [
            "",
            "=" * 66,
            f" EXPORT KOHYA  {self.root}",
            "=" * 66,
        ]
        if not self.per_class:
            lines += ["  Nada que exportar.", "=" * 66]
            return "\n".join(lines)

        lines.append("  CARPETA                             IMGS  REPEATS   POR EPOCA")
        for name in sorted(self.per_class):
            count = self.per_class[name]
            repeats = int(name.split("_", 1)[0])
            lines.append(
                f"    {name:32s} {count:5d}  {repeats:6d}  {repeats * count:10d}"
            )
        lines += [
            "",
            f"  Imagenes exportadas   {self.total}",
            f"  Vistas por epoca      {self.effective_steps}",
        ]
        if self.converted:
            lines.append(f"  Convertidas a RGB     {self.converted}")
        if self.skipped_license:
            lines.append(
                f"  LICENCIA NO APTA (omitidas) {len(self.skipped_license)}"
                "   <-- revisar procedencia"
            )
        if self.skipped_no_caption:
            lines.append(
                f"  SIN CAPTION (omitidas) {len(self.skipped_no_caption)}"
                "   <-- kohya las ignoraria"
            )
        if self.missing_source:
            lines.append(f"  Archivo no encontrado  {len(self.missing_source)}")

        lines += [
            "",
            "  Siguiente paso:",
            f"    kohya --train_data_dir {self.root}",
            "=" * 66,
        ]
        return "\n".join(lines)


def _normalize_and_copy(src: Path, dst: Path) -> bool:
    """Copia la imagen a `dst`, convirtiendo a RGB si hace falta.

    Devuelve True si hubo conversión. Los PNG con alfa y el TIFF suelto del
    dataset heredado se reescriben como JPEG: kohya los aceptaría, pero el
    canal alfa se interpreta de forma inconsistente entre versiones.
    """
    with Image.open(src) as img:
        if img.mode == "RGB" and src.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(src, dst)
            return False
        img.convert("RGB").save(dst, quality=95, subsampling=0)
        return True


def _mkdir_after_rmtree(out_dir: Path, attempts: int = 10) -> None:
    """Crea `out_dir` reintentando el `PermissionError` de Windows.

    En Windows, `rmtree` puede devolver antes de que el directorio deje de
    existir de verdad: si un antivirus, el indexador o el cliente de Drive
    mantienen un handle abierto, el borrado queda pendiente y el `mkdir`
    inmediato falla con `WinError 5`. Se reproduce corriendo `export` sobre un
    árbol de 511 imágenes recién escrito.

    Importa porque el `rmtree` ya ocurrió: sin reintento, `export` deja el árbol
    de entrenamiento **borrado** y aborta con un traceback, que es el peor
    resultado posible de un comando cuyo propósito es reconstruirlo.
    """
    delay = 0.1
    for attempt in range(attempts):
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 2.0)


def export_kohya(
    records: list[ImageRecord],
    source_dir: Path,
    out_dir: Path,
    repeats: dict[ImageClass, int] | None = None,
    write_csv: bool = True,
    clean: bool = True,
) -> ExportStats:
    """Escribe el árbol de entrenamiento a partir de los records activos."""
    repeats = repeats or CLASS_REPEATS
    source_dir, out_dir = Path(source_dir), Path(out_dir)
    stats = ExportStats(root=out_dir)

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    _mkdir_after_rmtree(out_dir)

    csv_rows: list[tuple[str, str]] = []

    for rec in records:
        if rec.rejected or rec.klass is ImageClass.UNCLASSIFIED:
            continue
        # Última barrera antes del entrenamiento. `recover` y `fetch` ya
        # aplican la política, pero este es el punto donde una imagen deja de
        # ser un candidato y pasa a ser material publicado: comprobarlo aquí
        # significa que ninguna ruta futura hacia el manifest —una fuente
        # nueva, un `restore` desafortunado, una edición a mano— puede colar
        # algo sin licencia en `dataset/train/`.
        if not is_allowed(rec.license):
            stats.skipped_license.append(rec.filename)
            continue
        if not rec.caption:
            stats.skipped_no_caption.append(rec.filename)
            continue

        src = source_dir / rec.filename
        if not src.exists():
            stats.missing_source.append(rec.filename)
            continue

        n = repeats.get(rec.klass, 1)
        folder = out_dir / f"{n}_{rec.klass.value}"
        folder.mkdir(parents=True, exist_ok=True)

        # Siempre .jpg: uniformar la extensión evita que kohya trate como
        # concepto distinto lo que sólo difiere en formato de archivo.
        dst = folder / (Path(rec.filename).stem + ".jpg")
        if _normalize_and_copy(src, dst):
            stats.converted += 1

        dst.with_suffix(".txt").write_text(rec.caption, encoding="utf-8")

        stats.per_class[folder.name] = stats.per_class.get(folder.name, 0) + 1
        csv_rows.append((dst.name, rec.caption))

    if write_csv and csv_rows:
        csv_path = out_dir / "captions_SDXL.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["file_name", "text"])
            writer.writerows(sorted(csv_rows))

    _write_readme(out_dir, stats)
    return stats


def _write_readme(out_dir: Path, stats: ExportStats) -> None:
    """Deja constancia de cómo se armó el árbol, junto al árbol mismo."""
    lines = [
        "# Dataset de entrenamiento — Códice Sintético",
        "",
        "Generado por `codice-scraper export`. No editar a mano.",
        "",
        f"Trigger word: `{TRIGGER}`",
        "",
        "El prefijo numérico de cada carpeta son las repeticiones por época que",
        "aplica kohya. Compensan el desbalance entre clases sin duplicar archivos:",
        "hay cientos de estratos y sólo decenas de plastiglomerados reales.",
        "",
        "| Carpeta | Imágenes | Repeats | Vistas por época |",
        "|---|---:|---:|---:|",
    ]
    for name in sorted(stats.per_class):
        count = stats.per_class[name]
        n = int(name.split("_", 1)[0])
        lines.append(f"| `{name}` | {count} | {n} | {n * count} |")
    lines += [
        f"| **Total** | **{stats.total}** | | **{stats.effective_steps}** |",
        "",
        "`captions_SDXL.csv` reproduce los captions con las cabeceras",
        "`file_name,text` que espera `02_SDXL_LoRA_Captions_Check.ipynb`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
