"""Hoja de contacto HTML para revisión visual.

Ninguna heurística sustituye a mirar el dataset. Esta hoja existe para que la
purga sea una decisión con las imágenes delante: un martillo de geólogo en una
esquina puede pasar, una cartela de museo no, y esa diferencia no la resuelve
un umbral.

Las miniaturas van embebidas como data URI para que el archivo sea autónomo y
se pueda mandar por WhatsApp al resto del equipo.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ..hashing import load_thumb
from ..models import ImageRecord

THUMB_PX = 240

_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:#12110f; color:#e8e4dc;
       font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; }
h1 { font-size:20px; margin:0 0 4px; letter-spacing:-.01em; }
.sub { color:#8f8779; margin-bottom:20px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px;
          padding:12px 16px; background:#1b1916; border-radius:8px; }
.legend b { color:#e8e4dc; }
.grid { display:grid; gap:14px;
        grid-template-columns:repeat(auto-fill,minmax(240px,1fr)); }
figure { margin:0; background:#1b1916; border-radius:8px; overflow:hidden;
         border:2px solid transparent; }
figure.ok { border-color:#2f6b45; }
figure.warn { border-color:#a3821f; }
figure.bad { border-color:#8a3232; opacity:.62; }
img { width:100%; height:190px; object-fit:cover; display:block; background:#000; }
figcaption { padding:8px 10px; font-size:11.5px; }
.name { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; color:#cfc7b8;
        word-break:break-all; }
.meta { color:#8f8779; margin-top:3px; }
.tags { margin-top:6px; display:flex; gap:4px; flex-wrap:wrap; }
.tag { font-size:10px; padding:1px 6px; border-radius:99px;
       background:#3a2020; color:#e0a0a0; }
.tag.info { background:#20303a; color:#9ec8e0; }
.tag.warn { background:#3a3020; color:#e0c89e; }
.cap { margin-top:6px; color:#a89f8e; font-style:italic; font-size:11px; }
"""


def _data_uri(path: Path, px: int = THUMB_PX) -> str | None:
    try:
        thumb = load_thumb(path, size=px)
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def write_contact_sheet(
    records: list[ImageRecord],
    source_dir: Path,
    out_path: Path,
    title: str = "Revisión del dataset",
    show_progress: bool = True,
) -> int:
    """Escribe la hoja de contacto. Devuelve cuántas miniaturas embebió."""
    source_dir, out_path = Path(source_dir), Path(out_path)

    rejected = sum(1 for r in records if r.rejected)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title>",
        f"<style>{_CSS}</style>",
        f"<h1>{html.escape(title)}</h1>",
        f"<div class='sub'>{len(records)} imágenes · "
        f"{len(records) - rejected} activas · {rejected} descartadas</div>",
        "<div class='legend'>"
        "<span><b style='color:#5aa87a'>Borde verde</b> = entra al entrenamiento</span>"
        "<span><b style='color:#c9a53f'>Borde amarillo</b> = entra, pero revisar</span>"
        "<span><b style='color:#c76b6b'>Borde rojo</b> = descartada</span>"
        "<span>Las etiquetas rojas son la razón del descarte; las amarillas, la duda</span>"
        "</div>",
        "<div class='grid'>",
    ]

    embedded = 0
    iterator = (
        tqdm(records, desc="  hoja de contacto", unit="img") if show_progress else records
    )
    for rec in iterator:
        uri = _data_uri(source_dir / rec.filename)
        if uri:
            embedded += 1

        if rec.rejected:
            css = "bad"
        elif rec.needs_review:
            css = "warn"
        else:
            css = "ok"
        dims = f"{rec.width}×{rec.height}" if rec.width else "?"
        lic = rec.license or "?"

        tags = "".join(
            f"<span class='tag'>{html.escape(r.value)}</span>" for r in rec.reject_reasons
        )
        tags += "".join(
            f"<span class='tag warn'>{html.escape(note)}</span>" for note in rec.needs_review
        )
        if rec.panoramic:
            tags += "<span class='tag info'>panorámica</span>"
        if rec.klass.value != "_unclassified":
            tags += f"<span class='tag info'>{html.escape(rec.klass.value)}</span>"

        cap = (
            f"<div class='cap'>{html.escape(rec.caption[:110])}</div>" if rec.caption else ""
        )
        img = (
            f"<img loading='lazy' src='{uri}' alt=''>"
            if uri
            else "<img alt='no se pudo leer'>"
        )

        parts.append(
            f"<figure class='{css}'>{img}<figcaption>"
            f"<div class='name'>{html.escape(rec.filename)}</div>"
            f"<div class='meta'>{dims} · {html.escape(lic)}</div>"
            f"<div class='tags'>{tags}</div>{cap}"
            "</figcaption></figure>"
        )

    parts.append("</div>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return embedded
