"""pipeline._download: despacha a `source.download()`, no siempre GET plano.

Antes de esto, `_download` siempre hacía un GET directo a `download_url` sin
importar la fuente, lo cual asume que "una URL = un archivo descargable". Esa
asunción es falsa para Europe PMC (las figuras vienen dentro de un ZIP por
artículo) y el fallo era silencioso: los descartes nunca llegaban al
manifest, así que no había ningún rastro de qué había fallado ni por qué.
"""

from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch

import pytest

from codice_scraper.models import ImageRecord
from codice_scraper.pipeline import _download, fetch
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


# --- limpieza del .part cuando el borrado en sí falla (Windows) -----------
#
# Incidente real: al bajar 555 candidatas reales en Windows, un solo archivo
# quedó bloqueado un instante (antivirus escaneando el .part recién escrito).
# El `except` de _download intentaba borrarlo sin capturar el fallo del
# borrado, así que el PermissionError del unlink reemplazaba silenciosamente
# la excepción original, escapaba de _download sin capturar, tumbaba el
# `ThreadPoolExecutor` completo y con él las ~140 descargas ya completadas de
# ese lote — nada llegó a manifest.save().


def test_download_no_revienta_si_falla_la_limpieza_del_part(tmp_path):
    source = _StubSource(error=ConnectionError("se cortó a medias"))
    rec = ImageRecord(filename="a.jpg", source="stub", download_url="x")

    part = tmp_path / "a.jpg.part"
    part.write_bytes(b"restos")

    with patch.object(Path, "unlink", side_effect=PermissionError(32, "bloqueado")):
        _, err = _download(rec, tmp_path, {"stub": source})

    # El motivo real de la descarga sigue siendo el que se reporta, no el
    # PermissionError de la limpieza fallida.
    assert err is not None and err.startswith("ConnectionError")


def test_fetch_no_pierde_el_lote_si_un_future_revienta_de_forma_inesperada(tmp_path):
    """Defensa en profundidad: aunque `_download` volviera a dejar escapar
    algo alguna vez, un solo fallo imprevisto no debe tirar las descargas ya
    completadas del resto del lote.
    """
    import codice_scraper.pipeline as pl
    from codice_scraper.models import ImageClass

    recs = [
        ImageRecord(
            filename=f"{name}.jpg",
            source="flaky",
            download_url=f"http://x/{name}",
            license="CC0",
            klass=ImageClass.ESTRATOS,
        )
        for name in ("bueno1", "malo", "bueno2")
    ]

    # `_download` real ya captura cualquier excepción de `source.download`,
    # así que para forzar el camino defensivo del `fetch()` se sustituye por
    # una versión que, para el record "malo", deja escapar la excepción tal
    # cual en vez de convertirla en (rec, error) — el escenario que la nueva
    # protección del bucle de recolección está pensada para cubrir.
    original_download = pl._download

    def flaky_download(rec, dest_dir, sources_by_name):
        if rec.filename == "malo.jpg":
            raise RuntimeError("se escapó de _download")
        return original_download(rec, dest_dir, sources_by_name)

    class DummySearchSource(Source):
        name = "flaky"

        def search(self, query, klass, limit=200):
            return iter(recs)

        def download(self, rec: ImageRecord) -> bytes:
            return b"contenido-bueno"

    with patch.object(pl, "_download", side_effect=flaky_download):
        from codice_scraper.manifest import Manifest

        manifest = Manifest(tmp_path / "m.jsonl")
        stats = fetch(
            [DummySearchSource()],
            {ImageClass.ESTRATOS: ["q"]},
            manifest,
            dest_dir=tmp_path / "incoming",
            limit_per_query=10,
        )

    assert stats.downloaded == 2, "las dos descargas buenas no deben perderse"
    assert stats.failed == 1
    assert manifest.get("bueno1.jpg") is not None
    assert manifest.get("bueno2.jpg") is not None
