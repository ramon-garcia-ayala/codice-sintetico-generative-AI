"""pipeline._download: despacha a `source.download()`, no siempre GET plano.

Antes de esto, `_download` siempre hacía un GET directo a `download_url` sin
importar la fuente, lo cual asume que "una URL = un archivo descargable". Esa
asunción es falsa para Europe PMC (las figuras vienen dentro de un ZIP por
artículo) y el fallo era silencioso: los descartes nunca llegaban al
manifest, así que no había ningún rastro de qué había fallado ni por qué.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codice_scraper.models import ImageRecord
from codice_scraper.pipeline import _download
from codice_scraper.sources.base import Source


class _StubSource(Source):
    name = "stub"

    def __init__(self, payload: bytes | None = None, error: Exception | None = None):
        super().__init__()
        self.payload = payload
        self.error = error
        self.calls = 0

    def search(self, query, klass, limit=200):
        return iter(())

    def download(self, rec: ImageRecord) -> bytes:
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


def test_download_usa_el_metodo_de_la_fuente(tmp_path):
    source = _StubSource(payload=b"bytes-reales")
    rec = ImageRecord(filename="a.jpg", source="stub", download_url="cualquier-cosa")

    out_rec, err = _download(rec, tmp_path, {"stub": source})

    assert err is None
    assert source.calls == 1
    assert (tmp_path / "a.jpg").read_bytes() == b"bytes-reales"


def test_download_propaga_el_error_con_su_tipo(tmp_path):
    source = _StubSource(error=FileNotFoundError("g002.jpg no está en el zip"))
    rec = ImageRecord(filename="a.jpg", source="stub", download_url="x")

    _, err = _download(rec, tmp_path, {"stub": source})

    assert err is not None
    assert err.startswith("FileNotFoundError")
    assert not (tmp_path / "a.jpg").exists()
    assert not (tmp_path / "a.jpg.part").exists()  # no deja restos a medias


def test_download_no_repite_si_el_archivo_ya_existe(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"ya-estaba")
    source = _StubSource(payload=b"esto-no-deberia-escribirse")
    rec = ImageRecord(filename="a.jpg", source="stub", download_url="x")

    _, err = _download(rec, tmp_path, {"stub": source})

    assert err is None
    assert source.calls == 0
    assert (tmp_path / "a.jpg").read_bytes() == b"ya-estaba"


def test_download_sin_fuente_registrada_cae_al_get_generico(tmp_path, monkeypatch):
    """Red de seguridad: si `rec.source` no está en el mapa, no debe reventar."""
    import codice_scraper.pipeline as pl

    class FakeResp:
        def raise_for_status(self):
            pass

        def iter_content(self, chunk):
            yield b"desde-http-generico"

    class FakeSession:
        def get(self, url, timeout, stream):
            return FakeResp()

    monkeypatch.setattr(pl, "http_session", lambda: FakeSession())
    rec = ImageRecord(filename="a.jpg", source="fuente-no-registrada", download_url="x")

    _, err = _download(rec, tmp_path, {})

    assert err is None
    assert (tmp_path / "a.jpg").read_bytes() == b"desde-http-generico"
