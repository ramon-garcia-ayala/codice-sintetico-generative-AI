"""Construcción de captions para kohya.

Un caption de LoRA no es una descripción literaria: es la lista de rasgos que
el modelo debe poder desacoplar del trigger. La estructura es siempre

    codice_geo, <clase>, <litología>, <estructura>, <color>, <encuadre>

El trigger va primero para que sea el ancla del concepto, y todo lo que no
queremos que quede pegado a él va después como atributo separable.
"""

from __future__ import annotations

import re

from .models import (
    CLASS_CAPTION_HINT,
    TRIGGER,
    ImageClass,
    ImageRecord,
)

#: Ruido habitual en las descripciones de Commons: nombres de cámara,
#: coordenadas, plantillas de wiki, créditos. No aporta nada visual.
_NOISE = re.compile(
    r"(\b\d{1,2}:\d{2}\b|\bDSC[_-]?\d+|\bIMG[_-]?\d+|\bcamera\b|\bnikon\b|\bcanon\b"
    r"|\bwikimedia\b|\bcommons\b|\bown work\b|\bself-photographed\b|\buploaded\b"
    r"|\bcc[- ]by[- ]sa\b|\bpublic domain\b|\bhttps?://\S+)",
    re.IGNORECASE,
)

#: Términos visuales que sí queremos conservar si aparecen. Mapea de la
#: categoría de Commons a un descriptor en inglés utilizable como tag.
_CATEGORY_TAGS = {
    "sandstone": "sandstone",
    "limestone": "limestone",
    "shale": "shale",
    "mudstone": "mudstone",
    "conglomerate": "conglomerate",
    "breccia": "breccia",
    "cross-bedding": "cross-bedded",
    "bedding": "horizontal bedding",
    "strata": "layered strata",
    "stratigraphy": "stratigraphic section",
    "outcrop": "rock outcrop",
    "cliff": "cliff face",
    "fold": "folded strata",
    "unconformit": "unconformity",
    "slag": "vitrified slag",
    "ignimbrite": "ignimbrite",
    "volcanic": "volcanic rock",
    "fulgurite": "fulgurite",
    "plastiglomerate": "fused plastic and sediment",
    "marine debris": "marine debris",
    "plastic": "embedded plastic fragments",
}

#: Descriptores de encuadre según la geometría de la imagen.
_FRAMING_PANORAMIC = "wide panoramic section"
_FRAMING_DEFAULT = "close detail"


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = _NOISE.sub(" ", text)
    text = re.sub(r"[^\w\s,.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip(" ,.-")


def _tags_from_categories(categories: list[str]) -> list[str]:
    """Traduce categorías de Commons a tags visuales, sin repetir."""
    joined = " ".join(categories).lower()
    tags: list[str] = []
    for needle, tag in _CATEGORY_TAGS.items():
        if needle in joined and tag not in tags:
            tags.append(tag)
    return tags


#: Debajo de esta saturación media la imagen es prácticamente monocroma. Más
#: de la mitad del dataset heredado lo es, porque la caliza y la roca gris
#: dominan; nombrarlo evita que el LoRA fije la desaturación dentro del
#: concepto en vez de tratarla como un atributo separable.
DESATURATED_THRESHOLD = 18.0


def build_caption(
    rec: ImageRecord,
    saturation: float | None = None,
    max_tags: int = 6,
) -> str:
    """Construye el caption de un record."""
    parts: list[str] = [TRIGGER]

    hint = CLASS_CAPTION_HINT.get(rec.klass)
    if hint:
        parts.append(hint)

    parts.extend(_tags_from_categories(rec.categories)[:max_tags])

    description = _clean(rec.description)
    if description:
        # Sólo la primera oración: lo que sigue suele ser contexto geográfico
        # o histórico, sin correlato visual.
        first = re.split(r"(?<=[.;])\s+", description)[0]
        words = first.split()
        if 2 <= len(words) <= 22:
            parts.append(first.lower().rstrip("."))

    if saturation is None:
        saturation = rec.saturation
    if saturation is not None and saturation < DESATURATED_THRESHOLD:
        parts.append("desaturated grey tones")

    parts.append(_FRAMING_PANORAMIC if rec.panoramic else _FRAMING_DEFAULT)

    # Dedup preservando orden, y recorte de vacíos.
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        key = part.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(part.strip())
    return ", ".join(out)


def caption_all(records: list[ImageRecord], overwrite: bool = False) -> int:
    """Rellena `caption` en los records que no lo tengan. Devuelve cuántos tocó."""
    n = 0
    for rec in records:
        if rec.caption and not overwrite:
            continue
        if rec.klass is ImageClass.UNCLASSIFIED:
            continue
        rec.caption = build_caption(rec)
        n += 1
    return n
