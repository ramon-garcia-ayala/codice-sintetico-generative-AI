"""Clasificación por categorías de Commons y construcción de captions."""

from __future__ import annotations

import pytest

from codice_scraper.captions import build_caption
from codice_scraper.classify import classify, classify_all, load_group_index
from codice_scraper.models import TRIGGER, ImageClass, ImageRecord, RejectReason


@pytest.fixture
def index() -> dict[str, str]:
    return load_group_index()


def _rec(**kw) -> ImageRecord:
    kw.setdefault("filename", "x.jpg")
    kw.setdefault("source", "wikimedia")
    return ImageRecord(**kw)


# --- clasificación --------------------------------------------------------


def test_el_yaml_de_categorias_carga(index):
    assert index["sedimentary rocks"] == "estratos"
    assert index["plastiglomerate"] == "plastiglomerado"
    assert index["fluorite"] == "legacy_noise"


def test_clasifica_estratos(index):
    rec = _rec(categories=["Sedimentary rocks", "Cross-bedding"])
    assert classify(rec, index) is ImageClass.ESTRATOS


def test_clasifica_proxy(index):
    assert classify(_rec(categories=["Slag"]), index) is ImageClass.PROXY


def test_plastiglomerado_gana_los_empates(index):
    """Es la clase escasa: perder una real dentro de estratos cuesta más."""
    rec = _rec(categories=["Sedimentary rocks", "Plastiglomerate"])
    assert classify(rec, index) is ImageClass.PLASTIGLOMERADO


def test_solo_mineral_de_coleccion_queda_sin_clasificar(index):
    """El caso de la fluorita en vitrina del dataset heredado."""
    rec = _rec(categories=["Fluorite", "Minerals"])
    assert classify(rec, index) is ImageClass.UNCLASSIFIED


def test_sin_categorias_no_se_adivina(index):
    assert classify(_rec(categories=[]), index) is ImageClass.UNCLASSIFIED


def test_categoria_desconocida_no_se_fuerza(index):
    assert classify(_rec(categories=["Cats"]), index) is ImageClass.UNCLASSIFIED


def test_classify_all_marca_el_mineral_de_coleccion(index):
    rec = _rec(categories=["Fluorite"])
    classify_all([rec], index)
    assert rec.rejected
    assert RejectReason.MANUAL in rec.reject_reasons
    assert any("mineral" in n for n in rec.needs_review)


def test_classify_all_no_marca_lo_simplemente_desconocido(index):
    rec = _rec(categories=["Cats"])
    classify_all([rec], index)
    assert not rec.rejected


def test_classify_all_respeta_lo_ya_clasificado(index):
    rec = _rec(categories=["Slag"], klass=ImageClass.ESTRATOS)
    classify_all([rec], index)
    assert rec.klass is ImageClass.ESTRATOS
    classify_all([rec], index, overwrite=True)
    assert rec.klass is ImageClass.PROXY


# --- captions -------------------------------------------------------------


def test_el_caption_empieza_por_el_trigger():
    rec = _rec(klass=ImageClass.ESTRATOS, categories=["Sandstone"])
    assert build_caption(rec).startswith(TRIGGER + ",")


def test_el_caption_traduce_categorias_a_tags():
    rec = _rec(klass=ImageClass.ESTRATOS, categories=["Sandstone", "Cross-bedding"])
    caption = build_caption(rec)
    assert "sandstone" in caption
    assert "cross-bedded" in caption


def test_las_sinteticas_llevan_su_propio_token():
    """Permite medir su efecto y retirarlas sin reentrenar desde cero."""
    caption = build_caption(_rec(klass=ImageClass.SYNTH))
    assert "codice_synth" in caption


def test_el_caption_limpia_el_ruido_de_commons():
    rec = _rec(
        klass=ImageClass.ESTRATOS,
        description="Own work by uploader, CC BY-SA https://example.com/x DSC_0421",
    )
    caption = build_caption(rec)
    for ruido in ("DSC", "https://", "CC BY-SA", "Own work"):
        assert ruido not in caption


def test_el_caption_marca_las_panoramicas():
    normal = build_caption(_rec(klass=ImageClass.ESTRATOS))
    pano = build_caption(_rec(klass=ImageClass.ESTRATOS, panoramic=True))
    assert "close detail" in normal
    assert "panoramic" in pano


def test_el_caption_nombra_la_desaturacion():
    """Más de la mitad del dataset es roca gris; si no se nombra, el LoRA la fija."""
    rec = _rec(klass=ImageClass.ESTRATOS, saturation=8.0)
    assert "desaturated" in build_caption(rec)
    assert "desaturated" not in build_caption(_rec(klass=ImageClass.ESTRATOS, saturation=60.0))


def test_el_caption_no_repite_terminos():
    rec = _rec(klass=ImageClass.ESTRATOS, categories=["Sandstone", "Sandstone"])
    partes = [p.strip() for p in build_caption(rec).split(",")]
    assert len(partes) == len(set(partes))


def test_descripcion_demasiado_larga_se_omite():
    rec = _rec(klass=ImageClass.ESTRATOS, description=" ".join(["palabra"] * 40))
    assert "palabra" not in build_caption(rec)
