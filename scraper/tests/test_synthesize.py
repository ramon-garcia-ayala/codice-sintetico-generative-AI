"""synthesize.py — sólo la parte que no necesita GPU.

`synthesize()` en sí depende de torch/diffusers y de una GPU CUDA real, así
que no se ejercita aquí (sería un test de integración caro y frágil). Lo que
sí se prueba es la lógica determinista alrededor: la composición del
caption/prompt y que coincida con el resto del pipeline de captions.
"""

from __future__ import annotations

import random

from codice_scraper.models import CLASS_CAPTION_HINT, SYNTH_TOKEN, ImageClass, TRIGGER
from codice_scraper.synthesize import build_description, build_synth_caption


def test_build_description_es_determinista():
    a = build_description(random.Random(42))
    b = build_description(random.Random(42))
    assert a == b


def test_build_description_varia_con_la_semilla():
    valores = {build_description(random.Random(i)) for i in range(20)}
    assert len(valores) > 1, "seeds distintas deben poder producir descripciones distintas"


def test_caption_empieza_con_trigger_y_token_synth():
    caption = build_synth_caption("una descripción cualquiera")
    assert caption.startswith(f"{TRIGGER}, {SYNTH_TOKEN},")


def test_caption_usa_el_mismo_hint_que_el_resto_del_pipeline():
    """Si CLASS_CAPTION_HINT cambia, este caption debe cambiar con él, no divergir."""
    caption = build_synth_caption("x")
    assert CLASS_CAPTION_HINT[ImageClass.SYNTH] in caption


def test_caption_incluye_la_descripcion_completa():
    desc = "dominated by fused plastic threads, beach sediment, close macro detail"
    assert desc in build_synth_caption(desc)
