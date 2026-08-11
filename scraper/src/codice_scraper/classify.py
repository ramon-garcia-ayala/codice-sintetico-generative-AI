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

#: Señales débiles de fotografía de espécimen/vitrina en el texto de Commons.
#: Se comprobaron sobre el dataset real, en dos rondas:
#:
#: - "specimen" en la descripción: 27 imágenes activas de estratos, no todas
#:   contaminación — hay cortes pulidos de roca (wackestone, travertino) que sí
#:   muestran textura sedimentaria genuina junto a series de minerales de
#:   colección mal categorizados (ej. 15 fotos de asfaltita etiquetada
#:   "Sedimentary rocks" en Commons, con fondo de estudio que el filtro de
#:   píxeles no detecta porque el espécimen llena el encuadre).
#: - "museum"/"博物館" en el filename o el título: apareció al curar proxies —
#:   una foto de vitrina con cartelas y reflejos de vidrio (fondo no uniforme,
#:   tampoco detectable por píxeles) junto a especímenes de museo fotografiados
#:   limpio, sin vitrina visible.
#:
#: En ambos casos la decisión no es automatizable de forma confiable: se marca
#: `needs_review` y se decide con la imagen delante, en la hoja de contacto.
_SPECIMEN_LANGUAGE = ("specimen",)
_MUSEUM_LANGUAGE = ("museum", "museo", "博物館", "vitrine", "vitrina", "display case")


def _flag_studio_language(rec: ImageRecord) -> None:
    description = (rec.description or "").lower()
    if any(term in description for term in _SPECIMEN_LANGUAGE):
        note = "descripción menciona 'specimen': revisar si es foto de estudio"
        if note not in rec.needs_review:
            rec.needs_review.append(note)

    # El nombre de un museo suele colarse en el filename o el título de
    # Commons aunque no aparezca en la descripción — a diferencia de
    # "specimen", que sí solía estar en la descripción.
    wide_text = " ".join([rec.filename, rec.origin_title or ""]).lower()
    if any(term in wide_text for term in _MUSEUM_LANGUAGE):
        note = "nombre menciona museo/vitrina: revisar si es foto de estudio"
        if note not in rec.needs_review:
            rec.needs_review.append(note)


def load_group_index(config_path: Path | None = None) -> dict[str, str]:
    """Categoría o término de búsqueda en minúsculas -> nombre del grupo.

    Indexa `wikimedia_categories` **y** `search_terms`. Falta el segundo
    bloque hacía que `classify(..., overwrite=True)` mandara a UNCLASSIFIED
    cualquier record descubierto por texto libre en vez de por categoría de
    Commons: `record_from_page`/`recover._apply_metadata` añaden el término de
    búsqueda que encontró la imagen a `rec.categories` (así es como
    `wm_anthropocene_coastal_dune.jpg` terminó con `anthropocene plastic
    sediment` en su lista), pero como ese término sólo vivía en `search_terms`
    y no en `wikimedia_categories`, no había ninguna entrada del índice que lo
    reconociera. `fetch --klass X` ya fija la clase correcta al descubrir, así
    que esto no importaba mientras nadie pidiera `--overwrite` — hasta que se
    pidió, y una imagen real de plastiglomerado (una de sólo dos) perdió su
    clase y su caption en la misma pasada.
    """
    config_path = config_path or (
        Path(__file__).resolve().parents[2] / "config" / "queries.yaml"
    )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    index: dict[str, str] = {}
    for block in ("wikimedia_categories", "search_terms"):
        for group, items in (data.get(block) or {}).items():
            for item in items or []:
                index[item.lower()] = group
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
        # El caption lleva el nombre de la clase incrustado
        # (`CLASS_CAPTION_HINT`), así que sobrevivir a un cambio de clase lo
        # convierte en una descripción falsa. `caption_all` se salta los
        # records que ya tienen texto, de modo que invalidarlo aquí es lo que
        # hace que un `classify --overwrite` seguido de `caption` regenere lo
        # que toca sin exigir `--overwrite` también en el segundo comando.
        if klass is not rec.klass:
            rec.caption = None
        rec.klass = klass
        counts[klass.value] += 1

        if not rec.rejected:
            _flag_studio_language(rec)

        # Una imagen que sólo aparece en categorías de mineral de colección es
        # la contaminación que ya detectó el filtro de fondo de estudio. Se
        # marca explícitamente para que la razón quede en el manifest.
        if klass is ImageClass.UNCLASSIFIED and rec.categories:
            groups = {index.get(c.lower().strip()) for c in rec.categories}
            # Exige al menos un `legacy_noise`: que todas las categorías sean
            # desconocidas significa que no sabemos qué es, no que sea mineral
            # de colección. Confundir ambos casos descartaría material válido.
            if "legacy_noise" in groups and groups <= {"legacy_noise", None}:
                # `OUT_OF_SCOPE`, no `MANUAL`: es un veredicto del clasificador,
                # y `restore` sólo debe poder revertir decisiones humanas. Con
                # `MANUAL` aquí, un `restore` sobre un lote de nombres
                # reintroduciría ruido de catálogo que nadie decidió admitir.
                rec.reject(RejectReason.OUT_OF_SCOPE)
                if "mineral de colección, fuera del concepto" not in rec.needs_review:
                    rec.needs_review.append("mineral de colección, fuera del concepto")

    return counts
