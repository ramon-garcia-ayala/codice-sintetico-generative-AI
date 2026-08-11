"""Tipos del dominio: una imagen candidata y su procedencia.

`ImageRecord` es el único objeto que cruza todas las etapas. Las fuentes lo
producen sin descargar nada (así `fetch --dry-run` puede reportar volumen y
licencias antes de gastar ancho de banda); el pipeline lo va enriqueciendo con
hashes, métricas y veredictos.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

#: Token de disparo del LoRA. Coincide con `codice_geo_v#.safetensors` del brief.
TRIGGER = "codice_geo"

#: Token extra para las variaciones generadas. Permite medir su efecto y
#: retirarlas del prompt sin reentrenar desde cero.
SYNTH_TOKEN = "codice_synth"


class ImageClass(StrEnum):
    """Las cuatro clases del dataset. El valor es el nombre de la carpeta."""

    ESTRATOS = "01_real_estratos"
    PLASTIGLOMERADO = "02_real_plastiglomerado"
    PROXY = "03_proxy_materiales"
    SYNTH = "04_synth_plastiglomerado"
    UNCLASSIFIED = "_unclassified"


#: Repeticiones de kohya por clase. Compensan el desbalance de tamaños sin
#: duplicar archivos en disco: el plastiglomerado real pesa más por imagen
#: porque hay muchas menos, y las sintéticas pesan poco a propósito.
CLASS_REPEATS: dict[ImageClass, int] = {
    ImageClass.ESTRATOS: 10,
    ImageClass.PLASTIGLOMERADO: 15,
    ImageClass.PROXY: 8,
    ImageClass.SYNTH: 5,
}

#: Fragmento de caption que describe cada clase, después del trigger.
CLASS_CAPTION_HINT: dict[ImageClass, str] = {
    ImageClass.ESTRATOS: "sedimentary strata",
    ImageClass.PLASTIGLOMERADO: "plastiglomerate",
    ImageClass.PROXY: "fused industrial conglomerate",
    ImageClass.SYNTH: f"{SYNTH_TOKEN}, plastiglomerate",
}

#: Licencia desconocida. Se marca explícitamente en vez de dejarla vacía para
#: que aparezca en los reportes y sea una decisión visible, no un descuido.
UNKNOWN_LICENSE = "UNKNOWN"


class RejectReason(StrEnum):
    """Por qué una imagen no entra al entrenamiento."""

    TOO_SMALL = "too_small"
    STUDIO_BACKGROUND = "studio_background"
    BLURRY = "blurry"
    DUPLICATE = "duplicate"
    EXTREME_ASPECT = "extreme_aspect"
    UNREADABLE = "unreadable"
    MANUAL = "manual"


class ImageRecord(BaseModel):
    """Una imagen candidata, desde que se descubre hasta que se exporta."""

    # --- identidad ---
    filename: str
    sha256: str | None = None
    phash: str | None = None

    # --- procedencia ---
    source: str
    source_id: str | None = None
    origin_title: str | None = None
    origin_url: str | None = None
    download_url: str | None = None
    license: str = UNKNOWN_LICENSE
    attribution: str | None = None
    description: str | None = None
    categories: list[str] = Field(default_factory=list)

    # --- técnico ---
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    bytes: int | None = None
    sharpness: float | None = None
    saturation: float | None = None

    # --- clasificación y veredicto ---
    klass: ImageClass = ImageClass.UNCLASSIFIED
    panoramic: bool = False
    rejected: bool = False
    reject_reasons: list[RejectReason] = Field(default_factory=list)
    needs_review: list[str] = Field(default_factory=list)
    caption: str | None = None

    @property
    def short_side(self) -> int | None:
        if self.width and self.height:
            return min(self.width, self.height)
        return None

    @property
    def aspect(self) -> float | None:
        if self.width and self.height:
            return max(self.width, self.height) / min(self.width, self.height)
        return None

    @property
    def has_provenance(self) -> bool:
        """True si sabemos de dónde salió y bajo qué licencia."""
        return self.license != UNKNOWN_LICENSE and bool(self.origin_title)

    def reject(self, reason: RejectReason) -> None:
        self.rejected = True
        if reason not in self.reject_reasons:
            self.reject_reasons.append(reason)
