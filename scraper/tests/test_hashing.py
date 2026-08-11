"""pHash, distancia de Hamming y varianza del laplaciano."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from codice_scraper.hashing import (
    hamming,
    laplacian_variance,
    load_thumb,
    phash,
    sha256_file,
)


@pytest.fixture
def noise() -> Image.Image:
    rng = np.random.default_rng(1234)
    return Image.fromarray(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8))


def test_phash_es_hex_de_64_bits(noise):
    h = phash(noise)
    assert len(h) == 16
    int(h, 16)  # no debe lanzar


def test_phash_identico_para_la_misma_imagen(noise):
    assert phash(noise) == phash(noise.copy())


def test_phash_estable_al_reescalar(noise):
    """El caso de uso real: el archivo local contra la miniatura de Commons."""
    reescalada = noise.resize((300, 200)).resize((600, 400))
    assert hamming(phash(noise), phash(reescalada)) <= 5


def test_phash_estable_al_recomprimir(tmp_path, noise):
    path = tmp_path / "q.jpg"
    noise.save(path, quality=60)
    with Image.open(path) as recomprimida:
        assert hamming(phash(noise), phash(recomprimida.convert("RGB"))) <= 8


def test_phash_distingue_imagenes_distintas():
    rng = np.random.default_rng(7)
    a = Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))
    b = Image.fromarray(rng.integers(0, 255, (300, 300, 3), dtype=np.uint8))
    assert hamming(phash(a), phash(b)) > 15


def test_hamming_simetrico_y_cero_consigo_mismo(noise):
    a, b = phash(noise), phash(noise.rotate(90))
    assert hamming(a, a) == 0
    assert hamming(a, b) == hamming(b, a)


def test_laplaciano_distingue_nitida_de_borrosa(noise):
    borrosa = noise.filter(ImageFilter.GaussianBlur(5))
    assert laplacian_variance(noise) > laplacian_variance(borrosa) * 5


def test_laplaciano_no_revienta_con_imagen_diminuta():
    assert laplacian_variance(Image.new("RGB", (2, 2))) == 0.0


def test_sha256_detecta_cambio_de_un_byte(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"contenido")
    b.write_bytes(b"contenidos")
    assert sha256_file(a) != sha256_file(b)
    assert sha256_file(a) == sha256_file(a)


def test_load_thumb_respeta_el_lado_maximo(tmp_path, noise):
    path = tmp_path / "grande.jpg"
    noise.save(path)
    thumb = load_thumb(path, size=128)
    assert max(thumb.size) <= 128
    assert thumb.mode == "RGB"
