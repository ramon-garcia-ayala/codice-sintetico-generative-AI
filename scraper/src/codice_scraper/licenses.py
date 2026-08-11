"""Política de licencias del dataset.

Vive aparte y se aplica en el pipeline, no en cada fuente, para que la regla
sea una sola y valga igual para Wikimedia, Europe PMC o lo que se añada
después. Una fuente que filtre por su cuenta acaba divergiendo.

Dos exclusiones, ambas deliberadas:

- **NC (no comercial)** — el proyecto se expone en una institución y puede
  tener difusión comercial. No podemos garantizar el uso no comercial.
- **ND (sin obra derivada)** — todo el pipeline es transformación: se entrena
  un modelo con la imagen y se generan derivados. Es justo lo que ND prohíbe.

Ojo con el orden: `"cc by" in "cc by-nc-nd"` es verdadero. Comprobar primero
lo permitido y luego lo prohibido deja pasar exactamente lo que se quería
bloquear — el fallo que motivó este módulo.
"""

from __future__ import annotations

from .models import UNKNOWN_LICENSE

#: Marcas de licencia libre aceptadas.
ALLOWED_MARKERS = (
    "cc0",
    "cc by",
    "cc-by",
    "ccby",
    "public domain",
    "publicdomain",
    "dominio público",
    "pd-",
    "no restrictions",
)

#: Cláusulas que descartan, se comprueban primero.
DENIED_CLAUSES = (
    "-nc",
    " nc",
    "noncommercial",
    "non-commercial",
    "-nd",
    " nd",
    "noderiv",
    "no deriv",
    "fair use",
    "all rights reserved",
    "copyright",
)


def is_allowed(license_: str | None) -> bool:
    """True si la licencia permite entrenar y generar derivados."""
    if not license_ or license_ == UNKNOWN_LICENSE:
        return False

    text = license_.lower().strip()
    if any(clause in text for clause in DENIED_CLAUSES):
        return False
    return any(marker in text for marker in ALLOWED_MARKERS)


def explain(license_: str | None) -> str:
    """Motivo legible del veredicto, para los reportes."""
    if not license_ or license_ == UNKNOWN_LICENSE:
        return "sin licencia identificada"
    text = license_.lower().strip()
    for clause in DENIED_CLAUSES:
        if clause in text:
            return f"cláusula no admitida ({clause.strip()})"
    if not any(marker in text for marker in ALLOWED_MARKERS):
        return "licencia no reconocida como libre"
    return "admitida"
