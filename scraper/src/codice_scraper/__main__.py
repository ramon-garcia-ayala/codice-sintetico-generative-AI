"""CLI de codice-scraper.

Flujo completo desde cero:

    codice-scraper ingest                      copia el Drive a _incoming y siembra el manifest
    codice-scraper audit --path dataset/_incoming --write-manifest
    codice-scraper recover                     rescata procedencia por pHash contra Wikimedia
    codice-scraper classify                    asigna clase según categorías de Commons
    codice-scraper caption                     construye los captions
    codice-scraper sheet                       hoja de contacto para revisión visual
    codice-scraper export                      árbol de entrenamiento para kohya
    codice-scraper report --attributions       estado del manifest y ATTRIBUTIONS.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

from . import paths
from .audit import audit_directory
from .filters import FilterConfig, iter_images
from .hashing import sha256_file
from .manifest import Manifest, write_attributions
from .models import ImageClass, ImageRecord, UNKNOWN_LICENSE


def _filter_config(args: argparse.Namespace) -> FilterConfig:
    overrides = {}
    if getattr(args, "min_short_side", None):
        overrides["min_short_side"] = args.min_short_side
    if getattr(args, "min_sharpness", None) is not None:
        overrides["min_sharpness"] = args.min_sharpness
    return FilterConfig(**overrides)


# --- audit ----------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    root = Path(args.path) if args.path else paths.DRIVE_DATASET
    if not root.exists():
        print(f"error: no existe {root}", file=sys.stderr)
        return 1

    manifest = Manifest(paths.MANIFEST_PATH).load()
    known = {r.filename: r for r in manifest.all()}

    print(f"Auditando {root} …")
    report, records = audit_directory(root, _filter_config(args), records=known)
    print(report.render())

    if args.write_manifest:
        for rec in records:
            manifest.upsert(rec)
        manifest.save()
        print(f"\nManifest actualizado: {paths.MANIFEST_PATH}  ({len(manifest)} entradas)")

    return 0


# --- ingest ---------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    """Copia las imágenes del Drive a `_incoming/` y siembra el manifest.

    El Drive nunca se modifica: se lee y se copia. Es idempotente — vuelve a
    copiar sólo lo que falte o haya cambiado de tamaño.
    """
    src = Path(args.source) if args.source else paths.DRIVE_DATASET
    if not src.exists():
        print(f"error: no existe {src}", file=sys.stderr)
        return 1

    paths.ensure_dirs()
    dst = paths.INCOMING_DIR
    images = iter_images(src)
    print(f"Origen  {src}\nDestino {dst}\n{len(images)} imágenes encontradas\n")

    copied = skipped = 0
    for path in tqdm(images, desc="  copiando", unit="img"):
        target = dst / path.name
        if target.exists() and target.stat().st_size == path.stat().st_size:
            skipped += 1
            continue
        shutil.copy2(path, target)
        copied += 1

    print(f"\n  copiadas {copied}, ya presentes {skipped}")

    manifest = Manifest(paths.MANIFEST_PATH).load()
    print("\nSembrando manifest (sha256 de cada archivo)…")
    for path in tqdm(iter_images(dst), desc="  hashing", unit="img"):
        rec = manifest.get(path.name) or ImageRecord(
            filename=path.name,
            source=args.source_name,
            klass=ImageClass.UNCLASSIFIED,
            license=UNKNOWN_LICENSE,
        )
        if not rec.sha256:
            rec.sha256 = sha256_file(path)
        manifest.upsert(rec)
    manifest.save()

    print(f"\nManifest: {paths.MANIFEST_PATH}  ({len(manifest)} entradas)")
    print(
        "\nSiguiente paso:\n"
        "  codice-scraper audit --path dataset/_incoming --write-manifest\n"
        "  codice-scraper recover --dry-run"
    )
    return 0


# --- recover --------------------------------------------------------------


def cmd_recover(args: argparse.Namespace) -> int:
    from .recover import recover_provenance

    manifest = Manifest(paths.MANIFEST_PATH).load()
    if not len(manifest):
        print("El manifest está vacío. Corre primero `codice-scraper ingest`.")
        return 1

    stats = recover_provenance(
        manifest,
        root=paths.INCOMING_DIR,
        dry_run=args.dry_run,
        limit_per_category=args.limit,
        categories=args.category or None,
    )
    print(stats.render())

    if not args.dry_run:
        manifest.save()
        print(f"\nManifest actualizado: {paths.MANIFEST_PATH}")
    else:
        print("\n(dry-run: no se escribió nada)")
    return 0


# --- fetch ----------------------------------------------------------------


def _load_queries() -> tuple[dict, dict]:
    """Devuelve (categorías, términos de texto) desde `queries.yaml`."""
    import yaml

    config = Path(__file__).resolve().parents[2] / "config" / "queries.yaml"
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    return data.get("wikimedia_categories") or {}, data.get("search_terms") or {}


def _load_secrets() -> dict:
    import yaml

    secrets = Path(__file__).resolve().parents[2] / "config" / "secrets.yaml"
    if not secrets.exists():
        return {}
    return yaml.safe_load(secrets.read_text(encoding="utf-8")) or {}


def cmd_fetch(args: argparse.Namespace) -> int:
    from .classify import GROUP_TO_CLASS
    from .pipeline import fetch
    from .sources import DEFAULT_SOURCES, build_source

    paths.ensure_dirs()
    categories, terms = _load_queries()
    secrets = _load_secrets()

    names = args.source or list(DEFAULT_SOURCES)
    sources = [build_source(n, secrets.get(n, {})) for n in names]

    # Wikimedia se consulta por categoría; el resto por texto libre.
    queries: dict = {}
    for group, klass in GROUP_TO_CLASS.items():
        if args.klass and klass.value != args.klass:
            continue
        items: list[str] = []
        if any(s.name == "wikimedia" for s in sources):
            items += [f"Category:{c}" for c in categories.get(group, [])]
        items += terms.get(group, [])
        if items:
            queries[klass] = items

    if not queries:
        print("No hay consultas que ejecutar con esos filtros.")
        return 1

    manifest = Manifest(paths.MANIFEST_PATH).load()
    stats = fetch(
        sources,
        queries,
        manifest,
        dest_dir=paths.INCOMING_DIR,
        limit_per_query=args.limit,
        dry_run=args.dry_run,
        allow_unknown_license=args.allow_unknown_license,
    )
    print(stats.render())

    if not args.dry_run:
        manifest.save()
        print(f"\nManifest actualizado: {paths.MANIFEST_PATH}  ({len(manifest)} entradas)")
        print("\nSiguiente:  codice-scraper caption  &&  codice-scraper sheet")
    return 0


# --- refilter ---------------------------------------------------------


def cmd_refilter(args: argparse.Namespace) -> int:
    """Reaplica el umbral de resolución sobre métricas ya guardadas.

    Sólo toca `too_small`; el resto de razones de descarte (fondo de estudio,
    duplicado…) se dejan como están. No re-descarga ni re-mide nada.
    """
    from .filters import reevaluate_min_short_side
    from .models import ImageClass

    manifest = Manifest(paths.MANIFEST_PATH).load()
    target = ImageClass(args.klass) if args.klass else None

    changed_in, changed_out = [], []
    for rec in manifest.all():
        if target is not None and rec.klass is not target:
            continue
        was_rejected = rec.rejected
        if reevaluate_min_short_side(rec, args.min_short_side):
            manifest.upsert(rec)
            (changed_out if was_rejected else changed_in).append(rec.filename)

    manifest.save()

    print(f"\n  Umbral aplicado: {args.min_short_side}px"
          + (f"  (sólo clase {target.value})" if target else "  (todas las clases)"))
    print(f"  Rescatadas (ahora entran)   {len(changed_out)}")
    print(f"  Recién descartadas          {len(changed_in)}")
    if changed_out:
        print("\nSiguiente:  codice-scraper caption  &&  codice-scraper export")
    return 0


# --- classify -------------------------------------------------------------


def cmd_classify(args: argparse.Namespace) -> int:
    from .classify import classify_all

    manifest = Manifest(paths.MANIFEST_PATH).load()
    if not len(manifest):
        print("Manifest vacío. Corre primero `codice-scraper ingest`.")
        return 1

    records = manifest.all()
    counts = classify_all(records, overwrite=args.overwrite)
    for rec in records:
        manifest.upsert(rec)
    manifest.save()

    print("\n  CLASIFICACIÓN")
    for klass in ImageClass:
        print(f"    {klass.value:28s} {counts.get(klass.value, 0):4d}")

    sin_clase = counts.get(ImageClass.UNCLASSIFIED.value, 0)
    if sin_clase:
        print(
            f"\n  {sin_clase} sin clasificar: no tienen categorías de Commons "
            "recuperadas.\n  Corre `recover` antes, o asígnalas a mano en la hoja "
            "de contacto."
        )
    return 0


# --- caption --------------------------------------------------------------


def cmd_caption(args: argparse.Namespace) -> int:
    from .captions import build_caption, caption_all

    manifest = Manifest(paths.MANIFEST_PATH).load()
    records = manifest.all()
    n = caption_all(records, overwrite=args.overwrite)
    for rec in records:
        manifest.upsert(rec)
    manifest.save()

    con_caption = [r for r in records if r.caption and not r.rejected]
    print(f"\n  Captions generados: {n}")
    print(f"  Con caption y activos: {len(con_caption)}")

    if con_caption:
        print("\n  MUESTRA")
        for rec in con_caption[:6]:
            print(f"    {rec.filename}")
            print(f"      {rec.caption}")
    return 0


# --- sheet ----------------------------------------------------------------


def cmd_sheet(args: argparse.Namespace) -> int:
    from .export import write_contact_sheet

    manifest = Manifest(paths.MANIFEST_PATH).load()
    records = manifest.all()
    if args.only_rejected:
        records = [r for r in records if r.rejected]
    if args.klass:
        records = [r for r in records if r.klass.value == args.klass]

    if not records:
        print("Nada que mostrar con ese filtro.")
        return 0

    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.REPORTS_DIR / (args.out or "contact_sheet.html")
    n = write_contact_sheet(records, paths.INCOMING_DIR, out, title=args.title)
    print(f"\n  {n} miniaturas embebidas en {out}")
    print("  Ábrela en el navegador y revisa antes de exportar.")
    return 0


# --- export ---------------------------------------------------------------


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_kohya

    manifest = Manifest(paths.MANIFEST_PATH).load()
    records = manifest.active()
    if not records:
        print("No hay imágenes activas en el manifest.")
        return 1

    out = Path(args.out) if args.out else paths.DATASET_DIR / "train"
    stats = export_kohya(records, paths.INCOMING_DIR, out, clean=not args.keep)
    print(stats.render())

    if stats.skipped_no_caption:
        print("  Corre `codice-scraper caption` para las que faltan.")
    return 0


# --- reject / restore -------------------------------------------------


def cmd_reject(args: argparse.Namespace) -> int:
    """Descarte manual tras revisión visual de la hoja de contacto."""
    from .models import RejectReason

    manifest = Manifest(paths.MANIFEST_PATH).load()
    note = f"revisión manual: {args.reason}" if args.reason else "revisión manual"

    touched, missing, already = [], [], []
    for name in args.filenames:
        rec = manifest.get(name)
        if rec is None:
            missing.append(name)
            continue
        if rec.rejected:
            already.append(name)
            continue
        rec.reject(RejectReason.MANUAL)
        if note not in rec.needs_review:
            rec.needs_review.append(note)
        manifest.upsert(rec)
        touched.append(name)

    manifest.save()

    print(f"\n  Descartadas ahora     {len(touched)}")
    for n in touched:
        print(f"    {n}")
    if already:
        print(f"  Ya estaban descartadas {len(already)}: {', '.join(already)}")
    if missing:
        print(f"  No encontradas en el manifest {len(missing)}: {', '.join(missing)}")

    if touched:
        print("\nSiguiente:  codice-scraper export")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    """Revierte un descarte manual (no toca descartes automáticos por filtro)."""
    from .models import RejectReason

    manifest = Manifest(paths.MANIFEST_PATH).load()
    touched, missing, not_manual = [], [], []
    for name in args.filenames:
        rec = manifest.get(name)
        if rec is None:
            missing.append(name)
            continue
        if RejectReason.MANUAL not in rec.reject_reasons:
            not_manual.append(name)
            continue
        rec.reject_reasons.remove(RejectReason.MANUAL)
        rec.rejected = bool(rec.reject_reasons)
        manifest.upsert(rec)
        touched.append(name)

    manifest.save()
    print(f"  Restauradas {len(touched)}: {', '.join(touched) or '-'}")
    if not_manual:
        print(f"  No eran descartes manuales (no tocadas): {', '.join(not_manual)}")
    if missing:
        print(f"  No encontradas: {', '.join(missing)}")
    return 0


# --- report ---------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    manifest = Manifest(paths.MANIFEST_PATH).load()
    if not len(manifest):
        print("Manifest vacío.")
        return 0

    records = manifest.all()
    active = manifest.active()

    print("")
    print("=" * 66)
    print(f" MANIFEST  {paths.MANIFEST_PATH}")
    print("=" * 66)
    print(f"  Entradas         {len(records)}")
    print(f"  Activas          {len(active)}")
    print(f"  Descartadas      {len(records) - len(active)}")

    by_class: dict[str, int] = {}
    for rec in active:
        by_class[rec.klass.value] = by_class.get(rec.klass.value, 0) + 1
    print("\n  POR CLASE")
    for klass in ImageClass:
        print(f"    {klass.value:28s} {by_class.get(klass.value, 0):4d}")

    by_license: dict[str, int] = {}
    for rec in active:
        by_license[rec.license] = by_license.get(rec.license, 0) + 1
    print("\n  POR LICENCIA")
    for lic, count in sorted(by_license.items(), key=lambda kv: -kv[1]):
        flag = "  <-- sin identificar" if lic == UNKNOWN_LICENSE else ""
        print(f"    {lic:38s} {count:4d}{flag}")

    with_prov = sum(1 for r in active if r.has_provenance)
    pct = 100 * with_prov // len(active) if active else 0
    print(f"\n  Con procedencia   {with_prov}/{len(active)}  ({pct}%)")

    reasons: dict[str, int] = {}
    for rec in records:
        for reason in rec.reject_reasons:
            reasons[reason.value] = reasons.get(reason.value, 0) + 1
    if reasons:
        print("\n  RAZONES DE DESCARTE")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:28s} {count:4d}")

    captioned = sum(1 for r in active if r.caption)
    print(f"\n  Con caption       {captioned}/{len(active)}")
    print("=" * 66)

    if args.attributions:
        n = write_attributions(manifest, paths.ATTRIBUTIONS_PATH)
        print(f"\nATTRIBUTIONS.md escrito con {n} entradas: {paths.ATTRIBUTIONS_PATH}")

    return 0


# --- entrypoint -----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codice-scraper",
        description="Dataset geológico para el LoRA de Códice Sintético",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="mide un directorio de imágenes y reporta")
    a.add_argument("--path", help="directorio a auditar (por defecto, el Drive)")
    a.add_argument("--min-short-side", type=int, help="resolución mínima del lado corto")
    a.add_argument("--min-sharpness", type=float, help="varianza mínima del laplaciano")
    a.add_argument("--write-manifest", action="store_true", help="persistir las métricas")
    a.set_defaults(func=cmd_audit)

    i = sub.add_parser("ingest", help="copia el Drive a _incoming y siembra el manifest")
    i.add_argument("--source", help="directorio de origen (por defecto, el Drive)")
    i.add_argument("--source-name", default="wikimedia-legacy", help="etiqueta de origen")
    i.set_defaults(func=cmd_ingest)

    r = sub.add_parser("recover", help="rescata procedencia por pHash contra Wikimedia")
    r.add_argument("--dry-run", action="store_true", help="no escribe el manifest")
    r.add_argument("--limit", type=int, default=600, help="máx. candidatas por categoría")
    r.add_argument("--category", action="append", help="categoría concreta (repetible)")
    r.set_defaults(func=cmd_recover)

    f = sub.add_parser("fetch", help="descubre y descarga imágenes nuevas")
    f.add_argument("--source", action="append", help="fuente concreta (repetible)")
    f.add_argument("--klass", help="limitar a una clase (nombre de carpeta)")
    f.add_argument("--limit", type=int, default=100, help="máx. por consulta")
    f.add_argument("--dry-run", action="store_true", help="sólo reporta, no descarga")
    f.add_argument(
        "--allow-unknown-license",
        action="store_true",
        help="aceptar imágenes sin licencia identificada (por defecto se omiten)",
    )
    f.set_defaults(func=cmd_fetch)

    rf = sub.add_parser(
        "refilter", help="reaplica el umbral de resolución sobre métricas ya guardadas"
    )
    rf.add_argument("--min-short-side", type=int, required=True)
    rf.add_argument("--klass", help="limitar a una clase (nombre de carpeta); si se omite, todas")
    rf.set_defaults(func=cmd_refilter)

    c = sub.add_parser("classify", help="asigna clase según categorías de Commons")
    c.add_argument("--overwrite", action="store_true", help="reclasificar lo ya clasificado")
    c.set_defaults(func=cmd_classify)

    cap = sub.add_parser("caption", help="construye los captions de kohya")
    cap.add_argument("--overwrite", action="store_true", help="regenerar los existentes")
    cap.set_defaults(func=cmd_caption)

    sh = sub.add_parser("sheet", help="hoja de contacto HTML para revisión visual")
    sh.add_argument("--out", help="nombre del archivo dentro de reports/")
    sh.add_argument("--title", default="Revisión del dataset", help="título de la hoja")
    sh.add_argument("--only-rejected", action="store_true", help="sólo las descartadas")
    sh.add_argument("--klass", help="filtrar por clase (nombre de carpeta)")
    sh.set_defaults(func=cmd_sheet)

    e = sub.add_parser("export", help="árbol de entrenamiento para kohya")
    e.add_argument("--out", help="directorio de salida (por defecto dataset/train)")
    e.add_argument("--keep", action="store_true", help="no borrar la salida anterior")
    e.set_defaults(func=cmd_export)

    rj = sub.add_parser("reject", help="descarta a mano tras revisión visual")
    rj.add_argument("filenames", nargs="+", help="nombres de archivo a descartar")
    rj.add_argument("--reason", help="motivo breve, queda anotado en needs_review")
    rj.set_defaults(func=cmd_reject)

    rs = sub.add_parser("restore", help="revierte un descarte manual (reject)")
    rs.add_argument("filenames", nargs="+", help="nombres de archivo a restaurar")
    rs.set_defaults(func=cmd_restore)

    rep = sub.add_parser("report", help="estado del manifest")
    rep.add_argument("--attributions", action="store_true", help="escribir ATTRIBUTIONS.md")
    rep.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
