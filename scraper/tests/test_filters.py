"""Umbrales de descarte.

Las imágenes sintéticas de aquí reproducen los casos que aparecieron de verdad
en `02_DATASET`: el mineral de colección sobre fondo negro con cartela y el
afloramiento fotografiado en campo.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from codice_scraper.filters import (
    FilterConfig,
    apply_filters,
    find_duplicates,
    is_studio_background,
    measure,
    reevaluate_min_short_side,
)
from codice_scraper.models import ImageRecord, RejectReason


def _save(tmp_path, img: Image.Image, name: str = "x.jpg"):
    path = tmp_path / name
    img.save(path, quality=95)
    return path


def _campo(w=2000, h=1500, seed=0) -> Image.Image:
    """Textura heterogénea que llega hasta los bordes: foto de yacimiento."""
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 190, (h // 10, w // 10, 3), dtype=np.uint8)
    return Image.fromarray(base).resize((w, h), Image.Resampling.LANCZOS)


def _vitrina(w=2000, h=1500, backdrop=0) -> Image.Image:
    """Objeto centrado sobre fondo infinito: foto de museo o estudio."""
    rng = np.random.default_rng(3)
    canvas = np.full((h, w, 3), backdrop, dtype=np.uint8)
    ih, iw = h // 3, w // 3
    canvas[ih : 2 * ih, iw : 2 * iw] = rng.integers(90, 200, (ih, iw, 3), dtype=np.uint8)
    return Image.fromarray(canvas)


# --- fondo de estudio -----------------------------------------------------


def test_detecta_fondo_negro_de_vitrina(tmp_path):
    m = measure(_save(tmp_path, _vitrina(backdrop=0)))
    assert is_studio_background(m)


def test_detecta_fondo_blanco_de_estudio(tmp_path):
    m = measure(_save(tmp_path, _vitrina(backdrop=255)))
    assert is_studio_background(m)


def test_no_marca_foto_de_campo(tmp_path):
    m = measure(_save(tmp_path, _campo()))
    assert not is_studio_background(m)


# --- resolución y aspecto -------------------------------------------------


def test_descarta_por_debajo_del_minimo(tmp_path):
    rec = ImageRecord(filename="p.jpg", source="test")
    apply_filters(rec, measure(_save(tmp_path, _campo(900, 700))))
    assert rec.rejected
    assert RejectReason.TOO_SMALL in rec.reject_reasons


def test_acepta_en_el_minimo_exacto(tmp_path):
    rec = ImageRecord(filename="ok.jpg", source="test")
    apply_filters(rec, measure(_save(tmp_path, _campo(1600, 1024))))
    assert RejectReason.TOO_SMALL not in rec.reject_reasons


def test_panoramica_se_marca_pero_no_se_descarta(tmp_path):
    """La sección madre del brief es 3:1: las panorámicas son material, no ruido."""
    rec = ImageRecord(filename="pano.jpg", source="test")
    apply_filters(rec, measure(_save(tmp_path, _campo(3600, 1200))))
    assert rec.panoramic
    assert RejectReason.EXTREME_ASPECT not in rec.reject_reasons


def test_aspecto_extremo_si_se_descarta(tmp_path):
    rec = ImageRecord(filename="tira.jpg", source="test")
    apply_filters(rec, measure(_save(tmp_path, _campo(6000, 1100))))
    assert RejectReason.EXTREME_ASPECT in rec.reject_reasons


# --- duplicados -----------------------------------------------------------


def test_duplicado_exacto_por_sha256():
    a = ImageRecord(filename="a.jpg", source="t", sha256="ff", phash="0" * 16)
    b = ImageRecord(filename="b.jpg", source="t", sha256="ff", phash="0" * 16)
    dupes = find_duplicates([a, b])
    assert dupes == {"b.jpg": "a.jpg"}


def test_duplicado_perceptual_por_phash():
    """Misma foto reescalada: el solape típico entre Wikimedia y USGS."""
    a = ImageRecord(filename="a.jpg", source="t", sha256="1", phash="ffffffffffffffff")
    b = ImageRecord(filename="b.jpg", source="t", sha256="2", phash="fffffffffffffffe")
    assert find_duplicates([a, b]) == {"b.jpg": "a.jpg"}


def test_imagenes_distintas_no_son_duplicados():
    a = ImageRecord(filename="a.jpg", source="t", sha256="1", phash="ffffffffffffffff")
    b = ImageRecord(filename="b.jpg", source="t", sha256="2", phash="0000000000000000")
    assert find_duplicates([a, b]) == {}


def test_sin_phash_no_se_empareja_por_accidente():
    a = ImageRecord(filename="a.jpg", source="t", sha256="1")
    b = ImageRecord(filename="b.jpg", source="t", sha256="2")
    assert find_duplicates([a, b]) == {}


# --- configuración --------------------------------------------------------


def test_umbral_de_resolucion_es_configurable(tmp_path):
    path = _save(tmp_path, _campo(1200, 900))
    cfg = FilterConfig(min_short_side=1200)
    rec = ImageRecord(filename="c.jpg", source="test")
    apply_filters(rec, measure(path, cfg), cfg)
    assert RejectReason.TOO_SMALL in rec.reject_reasons


def test_rgba_se_marca_para_normalizar(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (1400, 1400), (120, 100, 80, 255)).save(path)
    rec = ImageRecord(filename="a.png", source="test")
    apply_filters(rec, measure(path))
    assert any("RGB" in n for n in rec.needs_review)


def test_reevaluate_rescata_sin_re_medir():
    """El caso real: 186 imágenes de Europe PMC, 640px en vez de 1024px."""
    rec = ImageRecord(filename="a.jpg", source="t", width=800, height=700)
    rec.reject(RejectReason.TOO_SMALL)

    changed = reevaluate_min_short_side(rec, min_short_side=640)

    assert changed
    assert not rec.rejected
    assert RejectReason.TOO_SMALL not in rec.reject_reasons


def test_reevaluate_no_toca_otras_razones():
    """Fondo de estudio no depende de la resolución: debe sobrevivir intacto."""
    rec = ImageRecord(filename="a.jpg", source="t", width=800, height=700)
    rec.reject(RejectReason.TOO_SMALL)
    rec.reject(RejectReason.STUDIO_BACKGROUND)

    reevaluate_min_short_side(rec, min_short_side=640)

    assert rec.rejected  # sigue descartada, por fondo de estudio
    assert RejectReason.STUDIO_BACKGROUND in rec.reject_reasons
    assert RejectReason.TOO_SMALL not in rec.reject_reasons


def test_reevaluate_puede_volver_a_descartar_con_umbral_mas_estricto():
    rec = ImageRecord(filename="a.jpg", source="t", width=700, height=600)
    changed = reevaluate_min_short_side(rec, min_short_side=768)
    assert changed
    assert rec.rejected
    assert RejectReason.TOO_SMALL in rec.reject_reasons


def test_reevaluate_sin_cambio_devuelve_false():
    rec = ImageRecord(filename="a.jpg", source="t", width=2000, height=1800)
    assert not reevaluate_min_short_side(rec, min_short_side=640)
    assert not rec.rejected


def test_reevaluate_sin_dimensiones_no_hace_nada():
    rec = ImageRecord(filename="a.jpg", source="t")
    assert not reevaluate_min_short_side(rec, min_short_side=640)


def test_reject_no_duplica_razones():
    rec = ImageRecord(filename="a.jpg", source="t")
    rec.reject(RejectReason.BLURRY)
    rec.reject(RejectReason.BLURRY)
    assert rec.reject_reasons == [RejectReason.BLURRY]
