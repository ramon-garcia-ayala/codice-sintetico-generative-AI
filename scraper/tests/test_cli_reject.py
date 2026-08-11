"""`reject` / `restore` — descarte manual tras revisión visual.

Este es el mecanismo con el que un humano revierte el veredicto automático de
la hoja de contacto: `reject` marca, `restore` sólo revierte lo que él mismo
marcó, nunca un descarte del filtro automático.
"""

from __future__ import annotations

from codice_scraper.__main__ import build_parser, cmd_reject, cmd_restore
from codice_scraper.manifest import Manifest
from codice_scraper.models import ImageClass, ImageRecord, RejectReason


def _seed(path, **records) -> Manifest:
    m = Manifest(path)
    for name, kw in records.items():
        m.upsert(ImageRecord(filename=name, source="wikimedia", **kw))
    m.save()
    return m


def test_reject_marca_una_activa(tmp_path, monkeypatch):
    manifest_path = tmp_path / "m.jsonl"
    _seed(manifest_path, a=dict(klass=ImageClass.ESTRATOS))
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    args = build_parser().parse_args(["reject", "a", "--reason", "estudio"])
    assert cmd_reject(args) == 0

    rec = Manifest(manifest_path).load().get("a")
    assert rec.rejected
    assert RejectReason.MANUAL in rec.reject_reasons
    assert any("estudio" in n for n in rec.needs_review)


def test_reject_marca_tambien_sobre_una_ya_descartada_por_filtro(
    tmp_path, monkeypatch, capsys
):
    """La decisión humana se registra siempre, aunque un filtro se adelantara.

    Antes se salteaba este caso, y entonces un `refilter` que retirase la razón
    automática devolvía al entrenamiento una imagen que alguien había rechazado
    explícitamente, sin rastro en el manifest.
    """
    manifest_path = tmp_path / "m.jsonl"
    ya_mala = ImageRecord(filename="a", source="wikimedia")
    ya_mala.reject(RejectReason.STUDIO_BACKGROUND)
    m = Manifest(manifest_path)
    m.upsert(ya_mala)
    m.save()
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    cmd_reject(build_parser().parse_args(["reject", "a"]))
    assert "ahora también marcadas a mano" in capsys.readouterr().out

    rec = Manifest(manifest_path).load().get("a")
    assert RejectReason.STUDIO_BACKGROUND in rec.reject_reasons
    assert RejectReason.MANUAL in rec.reject_reasons


def test_reject_reporta_lo_que_no_existe(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "m.jsonl"
    _seed(manifest_path)
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    cmd_reject(build_parser().parse_args(["reject", "fantasma.jpg"]))
    assert "No encontradas" in capsys.readouterr().out


def test_restore_revierte_solo_lo_manual(tmp_path, monkeypatch):
    manifest_path = tmp_path / "m.jsonl"
    _seed(manifest_path, a=dict(klass=ImageClass.ESTRATOS))
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    cmd_reject(build_parser().parse_args(["reject", "a"]))
    cmd_restore(build_parser().parse_args(["restore", "a"]))

    rec = Manifest(manifest_path).load().get("a")
    assert not rec.rejected


def test_restore_no_toca_descarte_automatico(tmp_path, monkeypatch, capsys):
    manifest_path = tmp_path / "m.jsonl"
    automatico = ImageRecord(filename="a", source="wikimedia")
    automatico.reject(RejectReason.TOO_SMALL)
    m = Manifest(manifest_path)
    m.upsert(automatico)
    m.save()
    monkeypatch.setattr("codice_scraper.paths.MANIFEST_PATH", manifest_path)

    cmd_restore(build_parser().parse_args(["restore", "a"]))
    assert "No eran descartes manuales" in capsys.readouterr().out

    rec = Manifest(manifest_path).load().get("a")
    assert rec.rejected  # sigue descartada: too_small no era un reject manual
