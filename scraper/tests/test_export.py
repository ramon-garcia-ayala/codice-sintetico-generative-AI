"""Export al árbol de kohya."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image

from codice_scraper.export import export_kohya, write_contact_sheet
from codice_scraper.models import CLASS_REPEATS, ImageClass, ImageRecord, RejectReason


@pytest.fixture
def source(tmp_path):
    """Directorio con una imagen por clase, más un PNG con alfa."""
    src = tmp_path / "src"
    src.mkdir()
    rng = np.random.default_rng(0)
    for name in ("estrato.jpg", "plastic.jpg", "proxy.jpg", "synth.jpg"):
        arr = rng.integers(60, 200, (1200, 1600, 3), dtype=np.uint8)
        Image.fromarray(arr).save(src / name, quality=90)
    Image.new("RGBA", (1200, 1200), (100, 90, 80, 255)).save(src / "alfa.png")
    return src


def _rec(name, klass, caption="codice_geo, sedimentary strata") -> ImageRecord:
    return ImageRecord(filename=name, source="t", klass=klass, caption=caption)


def test_crea_una_carpeta_por_clase_con_sus_repeats(source, tmp_path):
    records = [
        _rec("estrato.jpg", ImageClass.ESTRATOS),
        _rec("plastic.jpg", ImageClass.PLASTIGLOMERADO),
        _rec("proxy.jpg", ImageClass.PROXY),
        _rec("synth.jpg", ImageClass.SYNTH),
    ]
    stats = export_kohya(records, source, tmp_path / "train")

    for klass, repeats in CLASS_REPEATS.items():
        folder = tmp_path / "train" / f"{repeats}_{klass.value}"
        assert folder.is_dir(), f"falta {folder.name}"
    assert stats.total == 4


def test_cada_imagen_lleva_su_txt(source, tmp_path):
    """Sin el .txt gemelo kohya entrena con caption vacío y no avisa."""
    records = [_rec("estrato.jpg", ImageClass.ESTRATOS, "codice_geo, sandstone")]
    export_kohya(records, source, tmp_path / "train")

    folder = tmp_path / "train" / f"{CLASS_REPEATS[ImageClass.ESTRATOS]}_{ImageClass.ESTRATOS.value}"
    assert (folder / "estrato.jpg").exists()
    assert (folder / "estrato.txt").read_text(encoding="utf-8") == "codice_geo, sandstone"


def test_el_png_con_alfa_se_convierte_a_rgb(source, tmp_path):
    records = [_rec("alfa.png", ImageClass.ESTRATOS)]
    stats = export_kohya(records, source, tmp_path / "train")

    assert stats.converted == 1
    folder = tmp_path / "train" / f"{CLASS_REPEATS[ImageClass.ESTRATOS]}_{ImageClass.ESTRATOS.value}"
    salida = folder / "alfa.jpg"
    assert salida.exists()
    with Image.open(salida) as img:
        assert img.mode == "RGB"


def test_omite_las_descartadas_y_las_sin_caption(source, tmp_path):
    malo = _rec("plastic.jpg", ImageClass.PLASTIGLOMERADO)
    malo.reject(RejectReason.STUDIO_BACKGROUND)
    sin_caption = _rec("proxy.jpg", ImageClass.PROXY, caption=None)

    stats = export_kohya(
        [_rec("estrato.jpg", ImageClass.ESTRATOS), malo, sin_caption],
        source,
        tmp_path / "train",
    )
    assert stats.total == 1
    assert stats.skipped_no_caption == ["proxy.jpg"]


def test_omite_las_sin_clasificar(source, tmp_path):
    stats = export_kohya(
        [_rec("estrato.jpg", ImageClass.UNCLASSIFIED)], source, tmp_path / "train"
    )
    assert stats.total == 0


def test_reporta_el_archivo_que_falta(source, tmp_path):
    stats = export_kohya(
        [_rec("fantasma.jpg", ImageClass.ESTRATOS)], source, tmp_path / "train"
    )
    assert stats.missing_source == ["fantasma.jpg"]


def test_csv_compatible_con_el_notebook_de_captions(source, tmp_path):
    """`02_SDXL_LoRA_Captions_Check.ipynb` valida exactamente estas cabeceras."""
    export_kohya(
        [_rec("estrato.jpg", ImageClass.ESTRATOS, "codice_geo, x")],
        source,
        tmp_path / "train",
    )
    with open(tmp_path / "train" / "captions_SDXL.csv", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["file_name", "text"]
    assert rows[1] == ["estrato.jpg", "codice_geo, x"]


def test_vistas_por_epoca_multiplica_por_los_repeats(source, tmp_path):
    stats = export_kohya(
        [
            _rec("estrato.jpg", ImageClass.ESTRATOS),
            _rec("plastic.jpg", ImageClass.PLASTIGLOMERADO),
        ],
        source,
        tmp_path / "train",
    )
    esperado = CLASS_REPEATS[ImageClass.ESTRATOS] + CLASS_REPEATS[ImageClass.PLASTIGLOMERADO]
    assert stats.effective_steps == esperado


def test_reexportar_no_acumula_restos(source, tmp_path):
    out = tmp_path / "train"
    export_kohya([_rec("estrato.jpg", ImageClass.ESTRATOS)], source, out)
    export_kohya([_rec("plastic.jpg", ImageClass.PLASTIGLOMERADO)], source, out)

    folder = out / f"{CLASS_REPEATS[ImageClass.ESTRATOS]}_{ImageClass.ESTRATOS.value}"
    assert not folder.exists(), "la clase anterior debió limpiarse"


def test_la_hoja_de_contacto_es_autonoma(source, tmp_path):
    """Sin rutas externas: se puede mandar por WhatsApp y se ve igual."""
    out = tmp_path / "sheet.html"
    n = write_contact_sheet(
        [_rec("estrato.jpg", ImageClass.ESTRATOS)], source, out, show_progress=False
    )
    assert n == 1
    html = out.read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in html
    assert "estrato.jpg" in html


def test_la_hoja_marca_visualmente_las_descartadas(source, tmp_path):
    malo = _rec("estrato.jpg", ImageClass.ESTRATOS)
    malo.reject(RejectReason.STUDIO_BACKGROUND)
    out = tmp_path / "sheet.html"
    write_contact_sheet([malo], source, out, show_progress=False)

    html = out.read_text(encoding="utf-8")
    assert "figure class='bad'" in html
    assert "studio_background" in html
