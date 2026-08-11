"""Variaciones sintéticas de plastiglomerado, vía SDXL + IP-Adapter local.

Existe porque el plastiglomerado real es la clase escasa del proyecto: tras
curar Europe PMC y Wikimedia quedaron sólo 2 fotos genuinas y de calidad (ver
`scraper/README.md`). El brief y el calendario de control ya contemplaban este
caso — "generar ~100 variaciones sintéticas con IA. Etiquetar sintéticas
separado de las reales al entrenar" — y `models.CLASS_REPEATS` /
`SYNTH_TOKEN` ya estaban preparados para recibirlas.

`torch` y `diffusers` son dependencias opcionales (extra `synth` en
pyproject.toml): el resto del paquete no las necesita, así que se importan
sólo dentro de las funciones de este módulo, no al nivel del archivo.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .models import CLASS_CAPTION_HINT, SYNTH_TOKEN, ImageClass, ImageRecord

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "ip-adapter_sdxl.bin"

NEGATIVE_PROMPT = (
    "deformed, ugly, low quality, blurry, watermark, text, cartoon, "
    "illustration, painting, 3d render, cgi"
)

#: Fragmentos que se combinan para variar la composición sin salirse del
#: concepto. Cada generación elige uno de cada grupo, así que 4×4×4 = 64
#: combinaciones posibles antes de repetir — de sobra para ~100 imágenes con
#: seeds distintas.
_COMPOSITIONS = (
    "close macro detail, top-down view",
    "angled view, raking side light",
    "cross-section cut, exposed interior texture",
    "wide view, several fragments scattered on a surface",
)
_MIXES = (
    "dominated by fused plastic threads, netting and rope",
    "dominated by mineral sediment with sparse plastic inclusions",
    "roughly equal mix of sand, pebbles and melted plastic",
    "heavily weathered, plastic barely distinguishable from the rock matrix",
)
_CONTEXTS = (
    "beach sediment, sand and pebbles",
    "volcanic basalt matrix",
    "coastal dune stratigraphy",
    "tidal debris accumulation",
)


def build_description(rng: random.Random) -> str:
    """Compone la descripción de una variación. Determinista para un `rng` dado."""
    comp = rng.choice(_COMPOSITIONS)
    mix = rng.choice(_MIXES)
    ctx = rng.choice(_CONTEXTS)
    return f"{mix}, {ctx}, {comp}"


def build_synth_caption(description: str) -> str:
    """Caption completo, coherente con `captions.build_caption` para esta clase."""
    hint = CLASS_CAPTION_HINT[ImageClass.SYNTH]
    return f"codice_geo, {hint}, {description}, photographic detail, natural lighting"


@dataclass
class SynthesisResult:
    generated: int = 0
    references_used: dict[str, int] = field(default_factory=dict)
    output_dir: Path | None = None

    def render(self) -> str:
        lines = [
            "",
            "=" * 66,
            " SÍNTESIS  plastiglomerado (SDXL + IP-Adapter)",
            "=" * 66,
            f"  Generadas       {self.generated}",
        ]
        if self.references_used:
            lines += ["", "  POR REFERENCIA"]
            for name, n in self.references_used.items():
                lines.append(f"    {name:40s} {n:4d}")
        lines.append("=" * 66)
        return "\n".join(lines)


def _load_pipeline(ip_adapter_scale: float):
    """Carga SDXL + IP-Adapter en GPU. Import perezoso: sólo se paga el costo
    de `torch`/`diffusers` cuando de verdad se va a generar algo."""
    import torch
    from diffusers import AutoPipelineForText2Image

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no hay GPU CUDA disponible — la síntesis local necesita una. "
            "Usa Image_workflows/04_SDXL_IP.ipynb en Colab en su lugar."
        )

    pipe = AutoPipelineForText2Image.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16
    ).to("cuda")
    pipe.load_ip_adapter(IP_ADAPTER_REPO, subfolder="sdxl_models", weight_name=IP_ADAPTER_WEIGHT)
    pipe.set_ip_adapter_scale(ip_adapter_scale)
    return pipe, torch


def synthesize(
    reference_dir: Path,
    output_dir: Path,
    count: int = 100,
    steps: int = 30,
    # El notebook original (04_SDXL_IP.ipynb) usa 0.6, pero probado sobre las
    # 2 referencias reales de este proyecto reproducía casi literal la
    # composición y el CONTEXTO de la foto —incluida la vitrina de museo de
    # una de las dos, con vidrio, pared y pedestal reproducidos con detalle—
    # en vez de transferir sólo el estilo del material. A 0.35 el prompt
    # (que ya especifica playa/duna/matriz volcánica, nunca museo) vuelve a
    # tener peso real sobre la composición.
    ip_adapter_scale: float = 0.35,
    seed: int = 7797676568,
    resolution: int = 1024,
    progress=None,
) -> tuple[SynthesisResult, list[ImageRecord]]:
    """Genera `count` variaciones y devuelve records listos para el manifest.

    No escribe el manifest: eso lo decide el llamador (permite revisar antes
    de comprometer, igual que el resto del pipeline).
    """
    from PIL import Image

    reference_dir, output_dir = Path(reference_dir), Path(output_dir)
    refs = sorted(p for p in reference_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not refs:
        raise FileNotFoundError(f"no hay imágenes de referencia en {reference_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipe, torch = _load_pipeline(ip_adapter_scale)

    result = SynthesisResult(output_dir=output_dir)
    records: list[ImageRecord] = []
    rng = random.Random(seed)

    iterator = range(count)
    if progress is not None:
        iterator = progress(iterator, total=count)

    for i in iterator:
        ref_path = refs[i % len(refs)]
        ref_img = Image.open(ref_path).convert("RGB")
        ref_img.thumbnail((resolution, resolution))

        description = build_description(rng)
        # El prompt de generación y el caption de entrenamiento son el mismo
        # texto a propósito: build_synth_caption ya antepone el trigger y el
        # hint de clase, así que reutilizarlo evita que generación y caption
        # describan cosas distintas.
        caption = build_synth_caption(description)
        prompt = caption

        run_seed = seed + i
        generator = torch.Generator("cuda").manual_seed(run_seed)

        image = pipe(
            prompt=prompt,
            ip_adapter_image=ref_img,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

        filename = f"synth_{i:03d}_{ref_path.stem}.jpg"
        image.save(output_dir / filename, quality=95)

        rec = ImageRecord(
            filename=filename,
            source="synthesis",
            source_id=str(run_seed),
            origin_title=f"Síntesis SDXL+IP-Adapter — seed {run_seed}, escala {ip_adapter_scale}",
            attribution=f"Generado a partir de {ref_path.name} (CC BY-SA 4.0, Wikimedia Commons)",
            # No es CC BY-SA de Commons, es una obra derivada de una imagen que
            # sí lo es. Se declara así (no "CC0" a secas) para que quede
            # trazable en ATTRIBUTIONS.md de dónde viene el material fuente.
            license="CC BY-SA 4.0 (síntesis derivada, no descargada de Commons)",
            description=description,
            categories=[],
            klass=ImageClass.SYNTH,
            width=image.width,
            height=image.height,
        )
        rec.caption = caption
        records.append(rec)

        result.generated += 1
        result.references_used[ref_path.name] = result.references_used.get(ref_path.name, 0) + 1

    return result, records
