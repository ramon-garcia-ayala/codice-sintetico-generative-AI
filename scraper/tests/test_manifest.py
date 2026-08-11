"""Persistencia del manifest e idempotencia.

La idempotencia es lo que permite reejecutar el scraper sin duplicar trabajo y
complementar el dataset de Carlos cuando aparezca material nuevo.
"""

from __future__ import annotations

from codice_scraper.manifest import Manifest, write_attributions
from codice_scraper.models import (
    UNKNOWN_LICENSE,
    ImageClass,
    ImageRecord,
    RejectReason,
)


def _rec(name: str, **kw) -> ImageRecord:
    kw.setdefault("source", "wikimedia")
    return ImageRecord(filename=name, **kw)


def test_ida_y_vuelta_conserva_los_campos(tmp_path):
    path = tmp_path / "m.jsonl"
    m = Manifest(path)
    m.upsert(
        _rec(
            "a.jpg",
            sha256="ab",
            phash="00ff00ff00ff00ff",
            license="CC BY-SA 4.0",
            attribution="Fulana",
            klass=ImageClass.ESTRATOS,
            categories=["Sedimentary rocks"],
            width=2000,
            height=1500,
        )
    )
    m.save()

    got = Manifest(path).load().get("a.jpg")
    assert got is not None
    assert got.license == "CC BY-SA 4.0"
    assert got.attribution == "Fulana"
    assert got.klass is ImageClass.ESTRATOS
    assert got.categories == ["Sedimentary rocks"]
    assert got.width == 2000


def test_upsert_reemplaza_en_vez_de_duplicar(tmp_path):
    path = tmp_path / "m.jsonl"
    m = Manifest(path)
    m.upsert(_rec("a.jpg", license=UNKNOWN_LICENSE))
    m.upsert(_rec("a.jpg", license="CC0"))
    m.save()

    reloaded = Manifest(path).load()
    assert len(reloaded) == 1
    assert reloaded.get("a.jpg").license == "CC0"


def test_guardar_dos_veces_no_crece(tmp_path):
    """Sin esto, reejecutar el scraper anexaría líneas repetidas al JSONL."""
    path = tmp_path / "m.jsonl"
    m = Manifest(path)
    m.upsert(_rec("a.jpg"))
    m.upsert(_rec("b.jpg"))
    m.save()
    primera = path.read_text(encoding="utf-8")

    Manifest(path).load().save()
    assert path.read_text(encoding="utf-8") == primera
    assert len(primera.strip().splitlines()) == 2


def test_lineas_corruptas_no_tumban_la_carga(tmp_path, capsys):
    path = tmp_path / "m.jsonl"
    path.write_text(
        _rec("a.jpg").model_dump_json() + "\n"
        "{ esto no es json\n"
        + _rec("b.jpg").model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    m = Manifest(path).load()
    assert len(m) == 2
    assert "ilegible" in capsys.readouterr().out


def test_manifest_inexistente_carga_vacio(tmp_path):
    assert len(Manifest(tmp_path / "nope.jsonl").load()) == 0


def test_active_excluye_los_descartados(tmp_path):
    m = Manifest(tmp_path / "m.jsonl")
    m.upsert(_rec("ok.jpg"))
    malo = _rec("malo.jpg")
    malo.reject(RejectReason.STUDIO_BACKGROUND)
    m.upsert(malo)
    assert [r.filename for r in m.active()] == ["ok.jpg"]


def test_indices_de_deduplicacion(tmp_path):
    m = Manifest(tmp_path / "m.jsonl")
    m.upsert(_rec("a.jpg", sha256="aa", phash="ff" * 8, download_url="http://x/a.jpg"))
    m.upsert(_rec("b.jpg", sha256="bb"))
    assert m.known_sha256() == {"aa", "bb"}
    assert m.known_phash() == {"ff" * 8: "a.jpg"}
    assert m.known_urls() == {"http://x/a.jpg"}


def test_has_provenance_exige_licencia_y_titulo():
    assert not _rec("a.jpg").has_provenance
    assert not _rec("a.jpg", license="CC0").has_provenance
    assert _rec("a.jpg", license="CC0", origin_title="File:X.jpg").has_provenance


def test_attributions_agrupa_y_lista_los_desconocidos(tmp_path):
    m = Manifest(tmp_path / "m.jsonl")
    m.upsert(_rec("a.jpg", license="CC0", origin_title="File:A.jpg", attribution="Autor"))
    m.upsert(_rec("sin.jpg", license=UNKNOWN_LICENSE))

    out = tmp_path / "ATTRIBUTIONS.md"
    assert write_attributions(m, out) == 1

    text = out.read_text(encoding="utf-8")
    assert "## wikimedia" in text
    assert "File:A.jpg" in text and "Autor" in text
    assert "Sin procedencia identificada" in text
    assert "sin.jpg" in text
