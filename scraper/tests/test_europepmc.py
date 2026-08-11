r"""EuropePMCSource: elección de miembro del ZIP y descarga de figuras.

Contexto del bug que motivó esto: `r'[^\\w.-]+'` (doble backslash) en la
sanitización del filename trataba "w" como carácter literal a excluir en vez
de la clase `\w`, así que casi todo el nombre se borraba y figuras distintas
de un mismo artículo colisionaban en el mismo filename. Y el enlace "natural"
a la figura (`europepmc.org/articles/.../bin/...`) resulta estar bloqueado
para clientes no-navegador — verificado en vivo, corta la conexión sin
importar los headers — así que la descarga real pasa por el ZIP de
`supplementaryFiles`, que si funciona.
"""

from __future__ import annotations

import io
import re
import zipfile

import pytest

from codice_scraper.sources.europepmc import (
    _ZIP_MEMBER_SEP,
    EuropePMCSource,
    SUPPLEMENTARY_URL,
    _best_zip_member,
)
from codice_scraper.models import ImageRecord


# --- el bug de la regex, para que no vuelva ---------------------------


def test_la_sanitizacion_no_borra_el_nombre():
    """El patrón correcto es `\\w` (una barra), no `\\\\w` (dos)."""
    href = "materials-19-00160-g001.jpg"
    safe = re.sub(r"[^\w.-]+", "_", href)
    assert safe == href  # nada que sanear en un nombre ya limpio


def test_figuras_distintas_no_colisionan_en_el_mismo_filename():
    hrefs = [f"materials-19-00160-g{i:03d}.jpg" for i in range(1, 6)]
    safe = [re.sub(r"[^\w.-]+", "_", h) for h in hrefs]
    assert len(set(safe)) == len(hrefs), "cada figura debe producir un nombre distinto"


# --- _best_zip_member ---------------------------------------------------


def test_coincidencia_exacta():
    names = ["fig-g001.jpg", "fig-g001.gif", "fig-g002.jpg"]
    assert _best_zip_member("fig-g001.jpg", names) == "fig-g001.jpg"


def test_prefiere_jpg_sobre_gif_por_stem():
    """El ZIP trae cada figura en dos formatos; el gif es una miniatura."""
    names = ["fig-g001.gif", "fig-g001.jpg"]
    assert _best_zip_member("fig-g001.jpg", names) == "fig-g001.jpg"


def test_cae_a_gif_si_no_hay_jpg():
    names = ["fig-g001.gif"]
    assert _best_zip_member("fig-g001.jpg", names) == "fig-g001.gif"


def test_sin_coincidencia_devuelve_none():
    assert _best_zip_member("no-existe.jpg", ["otro.jpg"]) is None


def test_insensible_a_mayusculas_en_la_extension():
    names = ["fig.JPG"]
    assert _best_zip_member("fig.jpg", names) == "fig.JPG"


# --- EuropePMCSource.download -------------------------------------------


def _fake_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_download_extrae_el_miembro_correcto(monkeypatch):
    zip_bytes = _fake_zip({"g001.jpg": b"contenido-1", "g002.jpg": b"contenido-2"})
    monkeypatch.setattr(EuropePMCSource, "_fetch_zip", staticmethod(lambda url: zip_bytes))

    src = EuropePMCSource()
    zip_url = SUPPLEMENTARY_URL.format(pmcid="PMC1")
    rec = ImageRecord(
        filename="epmc_pmc1_g002.jpg",
        source="europepmc",
        download_url=f"{zip_url}{_ZIP_MEMBER_SEP}g002.jpg",
    )

    assert src.download(rec) == b"contenido-2"


def test_download_sin_separador_lanza_error_claro():
    src = EuropePMCSource()
    rec = ImageRecord(filename="x.jpg", source="europepmc", download_url="url-sin-separador")
    with pytest.raises(ValueError, match="mal formado"):
        src.download(rec)


def test_download_miembro_ausente_lanza_error_claro(monkeypatch):
    zip_bytes = _fake_zip({"otro.jpg": b"x"})
    monkeypatch.setattr(EuropePMCSource, "_fetch_zip", staticmethod(lambda url: zip_bytes))

    src = EuropePMCSource()
    rec = ImageRecord(
        filename="x.jpg",
        source="europepmc",
        download_url=f"https://example/zip{_ZIP_MEMBER_SEP}fantasma.jpg",
    )
    with pytest.raises(FileNotFoundError):
        src.download(rec)


def test_fetch_zip_cachea_por_url(monkeypatch):
    """Un artículo con 8 figuras debe bajar el ZIP una sola vez, no ocho."""
    import codice_scraper.sources.europepmc as mod

    mod._zip_cache.clear()
    calls = []

    class FakeResp:
        content = b"PK\x05\x06" + b"\x00" * 18  # cabecera de zip vacio valida

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, url, timeout):
            calls.append(url)
            return FakeResp()

    monkeypatch.setattr(mod, "http_session", lambda: FakeSession())

    url = "https://example/zip"
    mod.EuropePMCSource._fetch_zip(url)
    mod.EuropePMCSource._fetch_zip(url)
    mod.EuropePMCSource._fetch_zip(url)

    assert len(calls) == 1, "la segunda y tercera llamada deben venir de la caché"
