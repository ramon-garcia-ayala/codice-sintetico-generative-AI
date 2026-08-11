"""Clasificación de imágenes en las cuatro clases del dataset.

La señal viene de las categorías de Wikimedia que `recover` devuelve: una
imagen en `Category:Cross-bedding` es un estrato y una en
`Category:Plastiglomerate` no lo es. Es más fiable que cualquier heurística
sobre los píxeles, porque son etiquetas puestas por gente que sabe de geología.

Lo que no tiene procedencia recuperada se queda sin clasificar a propósito.
Adivinar la clase de una imagen cuya licencia tampoco conocemos sería acumular
dos incertidumbres en una decisión que luego nadie puede auditar.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from .models import ImageClass, ImageRecord, RejectReason

#: Grupo de `queries.yaml` -> clase del dataset. `legacy_noise` no mapea a
#: ninguna: son las categorías de mineral de colección que el scrape original
#: arrastró y que no corresponden al concepto que se quiere entrenar.
GROUP_TO_CLASS = {
    "estratos": ImageClass.ESTRATOS,
    "plastiglomerado": ImageClass.PLASTIGLOMERADO,
    "proxy": ImageClass.PROXY,
}

#: Señal débil de fotografía de espécimen/estudio en la descripción de Commons.
#: Se comprobó sobre el dataset real: 27 imágenes activas contienen "specimen",
#: pero no todas son contaminación — hay cortes pulidos de roca (wackestone,
#: travertino) que sí muestran textura sedimentaria genuina junto a series de
#: minerales de colección mal categorizados (ej. una serie de 15 fotos de
#: asfaltita etiquetada "Sedimentary rocks" en Commons, con fondo de estudio
#: que el filtro de píxeles no detecta porque el espécimen llena el encuadre).
#: Por eso esto sólo marca `needs_review`, nunca descarta: la decisión final
#: es visual, en la hoja de contacto.
_SPECIMEN_LANGUAGE = ("specimen",)


def _flag_specimen_language(rec: ImageRecord) -> None:
    text = (rec.description or "").lower()
    if any(term in text for term in _SPECIMEN_LANGUAGE):
        note = "descripción menciona 'specimen': revisar si es foto de estudio"
        if note not in rec.needs_review:
            rec.needs_review.append(note)


def load_group_index(config_path: Path | None = None) -> dict[str, str]:
    """Categoría en minúsculas -> nombre del grupo."""
    config_path = config_path or (
        Path(__file__).resolve().parents[2] / "config" / "queries.yaml"
    )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for group, cats in (data.get("wikimedia_categories") or {}).items():
        for cat in cats or []:
            index[cat.lower()] = group
    return index


def classify(rec: ImageRecord, index: dict[str, str]) -> ImageClass:
    """Decide la clase de un record por sus categorías.

    Cuenta a qué grupo pertenece cada categoría y se queda con el mayoritario.
    El plastiglomerado gana los empates: es la clase escasa del proyecto y
    perder una imagen real dentro de "estratos" cuesta más que lo contrario.
    """
    if not rec.categories:
        return ImageClass.UNCLASSIFIED

    votes: Counter[str] = Counter()
    for cat in rec.categories:
        group = index.get(cat.lower().strip())
        if group:
            votes[group] += 1

    if not votes:
        return ImageClass.UNCLASSIFIED

    if "plastiglomerado" in votes:
        return ImageClass.PLASTIGLOMERADO

    top = votes.most_common(1)[0][0]
    if top == "legacy_noise":
        # Sólo categorías de mineral de colección: no es lo que se entrena.
        return ImageClass.UNCLASSIFIED

    return GROUP_TO_CLASS.get(top, ImageClass.UNCLASSIFIED)


def classify_all(
    records: list[ImageRecord],
    index: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Counter[str]:
    """Clasifica en bloque. Devuelve el conteo por clase resultante."""
    index = index if index is not None else load_group_index()
    counts: Counter[str] = Counter()

    for rec in records:
        if rec.klass is not ImageClass.UNCLASSIFIED and not overwrite:
            counts[rec.klass.value] += 1
            continue

        klass = classify(rec, index)
        rec.klass = klass
        counts[klass.value] += 1

        if not rec.rejected:
            _flag_specimen_language(rec)

        # Una imagen que sólo aparece en categorías de mineral de colección es
        # la contaminación que ya detectó el filtro de fondo de estudio. Se
        # marca explícitamente para que la razón quede en el manifest.
        if klass is ImageClass.UNCLASSIFIED and rec.categories:
            groups = {index.get(c.lower().strip()) for c in rec.categories}
            # Exige al menos un `legacy_noise`: que todas las categorías sean
            # desconocidas significa que no sabemos qué es, no que sea mineral
            # de colección. Confundir ambos casos descartaría material válido.
            if "legacy_noise" in groups and groups <= {"legacy_noise", None}:
                rec.reject(RejectReason.MANUAL)
                if "mineral de colección, fuera del concepto" not in rec.needs_review:
                    rec.needs_review.append("mineral de colección, fuera del concepto")

    return counts
