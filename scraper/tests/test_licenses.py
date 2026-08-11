"""Política de licencias.

El caso que motiva estos tests: `"cc by" in "cc by-nc-nd"` es verdadero, así
que un filtro por inclusión que no excluya primero las cláusulas prohibidas
admite justo lo que pretende bloquear. Pasó de verdad — 32 imágenes NC/ND se
colaron en una corrida antes de centralizar esta política.
"""

from __future__ import annotations

import pytest

from codice_scraper.licenses import explain, is_allowed
from codice_scraper.models import UNKNOWN_LICENSE


@pytest.mark.parametrize(
    "license_",
    [
        "CC0",
        "cc0 1.0",
        "CC BY 4.0",
        "cc by",
        "CC BY-SA 3.0",
        "CC BY-SA 4.0",
        "Public domain",
        "public domain",
        "PD-USGov",
        "No restrictions",
    ],
)
def test_admite_licencias_libres(license_):
    assert is_allowed(license_)


@pytest.mark.parametrize(
    "license_",
    [
        "cc by-nc",
        "cc by-nc-nd",
        "CC BY-NC-SA 4.0",
        "CC BY-ND 4.0",
        "cc-by-nc-nd/4.0",
        "noncommercial",
        "All rights reserved",
        "Fair use",
        "Copyright held by author",
    ],
)
def test_rechaza_nc_nd_y_copyright(license_):
    """Ninguna de estas debe pasar aunque contenga 'cc by' como subcadena."""
    assert not is_allowed(license_)


def test_rechaza_lo_desconocido_y_lo_vacio():
    assert not is_allowed(None)
    assert not is_allowed("")
    assert not is_allowed(UNKNOWN_LICENSE)


def test_nd_se_rechaza_porque_el_pipeline_transforma():
    """Entrenar un modelo y generar derivados es exactamente lo que ND prohíbe."""
    assert not is_allowed("CC BY-ND 4.0")
    assert "nd" in explain("CC BY-ND 4.0")


def test_explain_da_el_motivo():
    assert explain(None) == "sin licencia identificada"
    assert explain("CC0") == "admitida"
    assert "cláusula no admitida" in explain("cc by-nc")
    assert "no reconocida" in explain("Licencia rara de museo")


def test_es_insensible_a_mayusculas_y_espacios():
    assert is_allowed("  CC BY-SA 4.0  ")
    assert not is_allowed("  CC BY-NC 4.0  ")
