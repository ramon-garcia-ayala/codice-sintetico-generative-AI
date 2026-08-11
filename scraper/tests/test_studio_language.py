"""`_flag_studio_language` — señales débiles de foto de vitrina en el texto.

Dos casos reales encontrados curando el dataset:

- La serie de asfaltita ("specimen" en la descripción) — ver classify.py.
- Una foto de vitrina del Stephen Hui Geological Museum (HKU): fondo con
  cartelas y reflejo de vidrio, no detectable por el filtro de píxeles porque
  el fondo no es uniforme, pero el nombre del archivo delata el museo.

Ninguno de los dos se descarta automáticamente: ambos son señales débiles
(hay especímenes de museo fotografiados limpio, y descripciones con
"specimen" que sí son textura real) y la decisión final es visual.
"""

from __future__ import annotations

from codice_scraper.classify import _flag_studio_language
from codice_scraper.models import ImageRecord


def _rec(**kw) -> ImageRecord:
    kw.setdefault("filename", "x.jpg")
    kw.setdefault("source", "wikimedia")
    return ImageRecord(**kw)


def test_marca_specimen_en_la_descripcion():
    rec = _rec(description="This specimen appears to be a type of asphaltite.")
    _flag_studio_language(rec)
    assert any("specimen" in n for n in rec.needs_review)


def test_marca_museo_en_el_filename():
    rec = _rec(filename="wm_x_stephen_hui_geological_museum_exhibit.jpg")
    _flag_studio_language(rec)
    assert any("museo" in n for n in rec.needs_review)


def test_marca_museo_en_el_origin_title():
    rec = _rec(origin_title="File:Breccia from Burke Museum.jpg")
    _flag_studio_language(rec)
    assert any("museo" in n for n in rec.needs_review)


def test_no_marca_nada_sin_señales():
    rec = _rec(
        filename="wm_x_slag_field_photo.jpg",
        origin_title="File:Slag outcrop, field photo.jpg",
        description="A field photograph of vitrified slag near the smelter site.",
    )
    _flag_studio_language(rec)
    assert rec.needs_review == []


def test_puede_marcar_ambas_senales_a_la_vez():
    rec = _rec(
        filename="wm_x_geological_museum.jpg",
        description="A specimen displayed at the entrance.",
    )
    _flag_studio_language(rec)
    assert len(rec.needs_review) == 2


def test_no_duplica_la_nota_si_se_llama_dos_veces():
    rec = _rec(filename="wm_x_museum.jpg")
    _flag_studio_language(rec)
    _flag_studio_language(rec)
    assert len(rec.needs_review) == 1
