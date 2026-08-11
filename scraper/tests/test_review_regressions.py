"""Regresiones de la revisión de código.

Un test por defecto confirmado. Cada uno reproduce la secuencia concreta que
lo destapaba, no una versión abstracta: lo que hace que estos fallos sean
caros es que son silenciosos, así que el test tiene que ejercitar el camino
completo hasta el punto donde el daño se vuelve observable.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from codice_scraper.captions import caption_all
from codice_scraper.classify import classify_all, load_group_index
from codice_scraper.export import export_kohya
from codice_scraper.filters import apply_filters, measure, reevaluate_min_short_side
from codice_scraper.licenses import is_allowed
from codice_scraper.manifest import Manifest
from codice_scraper.models import (
    RECOMPUTED_REJECT_REASONS,
    ImageClass,
    ImageRecord,
    RejectReason,
)
from codice_scraper.recover import apply_license_policy
from codice_scraper.__main__ import build_parser, cmd_reject, cmd_restore


def _img(tmp_path, name="a.jpg", w=2000, h=1500, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 190, (max(h // 10, 2), max(w // 10, 2), 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(base).resize((w, h), Image.Resampling.LANCZOS).save(path, quality=90)
    return path


# --- 1. gate de licencias en el camino ingest -> recover -> export --------


@pytest.mark.parametrize("license_", ["GFDL", "Copyrighted free use", "CC BY-NC 4.0"])
def test_recover_marca_licencia_no_apta(license_):
    """Las heredadas del Drive nunca pasan por fetch, único gate anterior."""
    rec = ImageRecord(filename="a.jpg", source="wikimedia", license=license_)
    assert not apply_license_policy(rec)
    assert rec.rejected
    assert RejectReason.LICENSE_DENIED in rec.reject_reasons


def test_recover_no_marca_licencia_apta():
    rec = ImageRecord(filename="a.jpg", source="wikimedia", license="CC BY-SA 4.0")
    assert apply_license_policy(rec)
    assert not rec.rejected


def test_license_policy_es_reversible_e_idempotente():
    """Si una corrida posterior recupera mejor licencia, el descarte se retira."""
    rec = ImageRecord(filename="a.jpg", source="wikimedia", license="GFDL")
    apply_license_policy(rec)
    apply_license_policy(rec)
    assert rec.reject_reasons.count(RejectReason.LICENSE_DENIED) == 1

    rec.license = "CC0"
    assert apply_license_policy(rec)
    assert not rec.rejected
    assert RejectReason.LICENSE_DENIED not in rec.reject_reasons


def test_export_es_la_ultima_barrera_de_licencia(tmp_path):
    """Aunque algo llegue activo al manifest, no debe entrar a dataset/train."""
    src = tmp_path / "src"
    src.mkdir()
    _img(src, "mala.jpg")
    rec = ImageRecord(
        filename="mala.jpg",
        source="x",
        klass=ImageClass.ESTRATOS,
        caption="codice_geo, sedimentary strata",
        license="CC BY-NC-ND 4.0",
    )
    stats = export_kohya([rec], src, tmp_path / "train")
    assert stats.total == 0
    assert stats.skipped_license == ["mala.jpg"]


# --- 2. colisión de nombres -----------------------------------------------


def test_wikimedia_incluye_pageid_para_no_colisionar():
    from codice_scraper.sources.wikimedia import record_from_page

    def page(pageid, title):
        return {
            "pageid": pageid,
            "title": title,
            "imageinfo": [
                {
                    "url": f"https://upload.example/{pageid}.jpg",
                    "descriptionurl": "https://commons.example",
                    "width": 2000,
                    "height": 1500,
                    "extmetadata": {"LicenseShortName": {"value": "CC BY-SA 4.0"}},
                }
            ],
        }

    a = record_from_page(page(11, "File:Cliff, Spain.jpg"), ImageClass.ESTRATOS, "x")
    b = record_from_page(page(22, "File:Cliff Spain.jpg"), ImageClass.ESTRATOS, "x")
    assert a.filename != b.filename, "títulos distintos deben dar nombres distintos"


def test_unique_filename_desambigua_solo_ante_colision_real(tmp_path):
    from codice_scraper.pipeline import _unique_filename

    manifest = Manifest(tmp_path / "m.jsonl")
    existing = ImageRecord(
        filename="wm_x.jpg", source="wikimedia", download_url="http://a/1.jpg"
    )
    manifest.upsert(existing)

    # Misma URL -> es la misma imagen, debe reutilizar el nombre (idempotencia).
    same = ImageRecord(
        filename="wm_x.jpg", source="wikimedia", download_url="http://a/1.jpg"
    )
    assert _unique_filename(same, set(), manifest, set()) == "wm_x.jpg"

    # URL distinta con el mismo nombre -> colisión, hay que desambiguar.
    other = ImageRecord(
        filename="wm_x.jpg", source="wikimedia", download_url="http://a/2.jpg"
    )
    got = _unique_filename(other, set(), manifest, set())
    assert got != "wm_x.jpg"
    assert got.endswith(".jpg")


def test_unique_filename_es_determinista(tmp_path):
    """Reejecutar el scraper debe producir el mismo nombre, no uno nuevo."""
    from codice_scraper.pipeline import _unique_filename

    manifest = Manifest(tmp_path / "m.jsonl")
    manifest.upsert(
        ImageRecord(filename="wm_x.jpg", source="w", download_url="http://a/1.jpg")
    )
    rec = ImageRecord(filename="wm_x.jpg", source="w", download_url="http://a/2.jpg")
    assert _unique_filename(rec, set(), manifest, set()) == _unique_filename(
        rec, set(), manifest, set()
    )


# --- 3. reject sobre imágenes ya descartadas ------------------------------


def test_reject_marca_aunque_un_filtro_ya_la_hubiera_descartado(tmp_path, monkeypatch):
    """El fallo completo: reject silencioso + refilter que la reintroduce."""
    manifest_path = tmp_path / "m.jsonl"
    manifest = Manifest(manifest_path)
    rec = ImageRecord(
        filename="a.jpg", source="w", klass=ImageClass.ESTRATOS, width=800, height=700
    )
    rec.reject(RejectReason.TOO_SMALL)
    manifest.upsert(rec)
    manifest.save()
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    cmd_reject(build_parser().parse_args(["reject", "a.jpg", "--reason", "vitrina"]))

    saved = Manifest(manifest_path).load().get("a.jpg")
    assert RejectReason.MANUAL in saved.reject_reasons
    assert any("vitrina" in n for n in saved.needs_review)

    # Y ahora el rescate por umbral no debe devolverla al entrenamiento.
    reevaluate_min_short_side(saved, min_short_side=640)
    assert saved.rejected, "la decisión humana debe sobrevivir al refilter"
    assert RejectReason.MANUAL in saved.reject_reasons


def test_reject_repetido_no_duplica_la_marca(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "m.jsonl"
    m = Manifest(manifest_path)
    m.upsert(ImageRecord(filename="a.jpg", source="w"))
    m.save()
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    cmd_reject(build_parser().parse_args(["reject", "a.jpg"]))
    cmd_reject(build_parser().parse_args(["reject", "a.jpg"]))
    assert "Ya estaban marcadas a mano" in capsys.readouterr().out

    saved = Manifest(manifest_path).load().get("a.jpg")
    assert saved.reject_reasons.count(RejectReason.MANUAL) == 1


# --- 4. razones obsoletas tras cambiar umbrales ---------------------------


def test_apply_filters_limpia_razones_que_ya_no_aplican(tmp_path):
    """Bajar el umbral debe tener efecto, no dejar el veredicto viejo pegado."""
    from codice_scraper.filters import FilterConfig

    path = _img(tmp_path, w=900, h=800)
    rec = ImageRecord(filename=path.name, source="w")

    apply_filters(rec, measure(path), FilterConfig(min_short_side=1024))
    assert RejectReason.TOO_SMALL in rec.reject_reasons

    cfg = FilterConfig(min_short_side=640)
    apply_filters(rec, measure(path, cfg), cfg)
    assert RejectReason.TOO_SMALL not in rec.reject_reasons
    assert not rec.rejected


def test_apply_filters_preserva_las_razones_que_no_son_de_pixel(tmp_path):
    path = _img(tmp_path)
    rec = ImageRecord(filename=path.name, source="w")
    rec.reject(RejectReason.MANUAL)
    rec.reject(RejectReason.LICENSE_DENIED)
    rec.reject(RejectReason.DUPLICATE)

    apply_filters(rec, measure(path))

    for reason in (
        RejectReason.MANUAL,
        RejectReason.LICENSE_DENIED,
        RejectReason.DUPLICATE,
    ):
        assert reason in rec.reject_reasons
        assert reason not in RECOMPUTED_REJECT_REASONS
    assert rec.rejected


def test_apply_filters_no_acumula_la_nota_de_rgb(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (1400, 1400), (120, 100, 80, 255)).save(path)
    rec = ImageRecord(filename="a.png", source="w")

    for _ in range(3):
        apply_filters(rec, measure(path))

    assert len([n for n in rec.needs_review if "RGB" in n]) == 1


def test_apply_filters_reinicia_panoramic(tmp_path):
    from codice_scraper.filters import FilterConfig

    path = _img(tmp_path, w=2000, h=1500)
    rec = ImageRecord(filename=path.name, source="w", panoramic=True)
    apply_filters(rec, measure(path), FilterConfig())
    assert not rec.panoramic, "1.33:1 no es panorámica"


# --- 5. restore no debe revertir veredictos automáticos -------------------


def test_classify_usa_out_of_scope_no_manual():
    index = load_group_index()
    rec = ImageRecord(filename="a.jpg", source="w", categories=["Fluorite"])
    classify_all([rec], index)

    assert RejectReason.OUT_OF_SCOPE in rec.reject_reasons
    assert RejectReason.MANUAL not in rec.reject_reasons


def test_restore_no_revierte_el_veredicto_del_clasificador(tmp_path, monkeypatch, capsys):
    index = load_group_index()
    rec = ImageRecord(filename="a.jpg", source="w", categories=["Fluorite"])
    classify_all([rec], index)

    manifest_path = tmp_path / "m.jsonl"
    m = Manifest(manifest_path)
    m.upsert(rec)
    m.save()
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    cmd_restore(build_parser().parse_args(["restore", "a.jpg"]))
    assert "No eran descartes manuales" in capsys.readouterr().out

    saved = Manifest(manifest_path).load().get("a.jpg")
    assert saved.rejected


# --- 6. captions obsoletos tras reclasificar ------------------------------


def test_classify_overwrite_invalida_el_caption_de_la_clase_anterior():
    index = load_group_index()
    rec = ImageRecord(
        filename="a.jpg",
        source="w",
        klass=ImageClass.ESTRATOS,
        categories=["Plastiglomerate"],
        caption="codice_geo, sedimentary strata, close detail",
    )

    classify_all([rec], index, overwrite=True)
    assert rec.klass is ImageClass.PLASTIGLOMERADO
    assert rec.caption is None, "el caption nombraba la clase anterior"

    caption_all([rec])
    assert "plastiglomerate" in rec.caption
    assert "sedimentary strata" not in rec.caption


def test_classify_sin_cambio_de_clase_conserva_el_caption():
    index = load_group_index()
    original = "codice_geo, sedimentary strata, sandstone, close detail"
    rec = ImageRecord(
        filename="a.jpg",
        source="w",
        klass=ImageClass.ESTRATOS,
        categories=["Sandstone"],
        caption=original,
    )
    classify_all([rec], index, overwrite=True)
    assert rec.caption == original


# --- 7. descargas ilegibles quedan registradas ----------------------------


def test_fetch_marca_las_ilegibles_en_vez_de_borrarlas(tmp_path, monkeypatch):
    """Sin record, su URL no entra en known_urls y se re-descarga para siempre."""
    import codice_scraper.pipeline as pl
    from codice_scraper.sources.base import Source

    dest = tmp_path / "incoming"
    dest.mkdir()

    class BrokenSource(Source):
        name = "broken"

        def search(self, query, klass, limit=200):
            yield ImageRecord(
                filename="roto.jpg",
                source="broken",
                download_url="http://x/roto.jpg",
                license="CC0",
                klass=ImageClass.ESTRATOS,
            )

        def download(self, rec):
            return b"esto no es un jpeg"

    manifest = Manifest(tmp_path / "m.jsonl")
    stats = pl.fetch(
        [BrokenSource()],
        {ImageClass.ESTRATOS: ["q"]},
        manifest,
        dest_dir=dest,
        limit_per_query=1,
    )

    assert stats.unreadable == 1
    saved = manifest.get("roto.jpg")
    assert saved is not None, "el record debe quedar, o la URL se re-descarga siempre"
    assert RejectReason.UNREADABLE in saved.reject_reasons
    assert "http://x/roto.jpg" in manifest.known_urls()
