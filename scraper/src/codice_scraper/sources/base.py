"""Interfaz común de las fuentes de imágenes.

Una `Source` **descubre**, no descarga. Devuelve `ImageRecord` con la URL y la
licencia ya resueltas, y es `pipeline.py` quien decide qué bajar. Esa
separación es lo que permite `fetch --dry-run`: saber cuántas imágenes hay y
bajo qué licencias antes de gastar un byte de ancho de banda.

Añadir una fuente es implementar `search()`. Nada más del paquete necesita
cambiar.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterator

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..models import ImageClass, ImageRecord

_STREAM_CHUNK = 1 << 16

#: Identificarse es requisito explícito en los términos de uso de la API de
#: Wikimedia y buena práctica en el resto.
USER_AGENT = (
    "codice-sintetico-dataset/0.1 "
    "(https://github.com/codice-sintetico; proyecto Fondo Creativo 2026) "
    "python-requests"
)

_local = threading.local()


def http_session() -> requests.Session:
    """Sesión con reintentos y backoff, una por hilo.

    `requests.Session` no es segura de compartir entre hilos y el pool por
    defecto se queda corto en cuanto se paraleliza: sin esto, una tanda
    concurrente contra Wikimedia falla en masa.
    """
    session = getattr(_local, "session", None)
    if session is not None:
        return session

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "image/*,*/*"})
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    _local.session = session
    return session


class Source(ABC):
    """Fuente de imágenes candidatas."""

    #: Identificador corto, se guarda en `ImageRecord.source`.
    name: str = "base"

    #: Si la fuente necesita credenciales y no las tiene, queda deshabilitada
    #: en vez de reventar a mitad de una corrida larga.
    requires_key: bool = False

    #: Fuentes cuya licencia no permite entrenamiento sin revisión previa se
    #: marcan aquí y quedan detrás de un flag explícito en la CLI.
    needs_explicit_optin: bool = False

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @property
    def available(self) -> bool:
        """False si le faltan credenciales."""
        return not self.requires_key or bool(self.config.get("api_key"))

    @abstractmethod
    def search(
        self, query: str, klass: ImageClass, limit: int = 200
    ) -> Iterator[ImageRecord]:
        """Devuelve candidatas para `query`, sin descargar los archivos."""
        raise NotImplementedError

    def download(self, rec: ImageRecord) -> bytes:
        """Obtiene los bytes de la imagen de `rec`.

        Implementación por defecto: un GET simple a `rec.download_url`. Una
        fuente cuyo backend de imágenes no sea "una URL, un archivo" —como
        Europe PMC, donde las figuras vienen empaquetadas en un ZIP por
        artículo— sobreescribe esto en vez de forzar ese caso al pipeline
        genérico.
        """
        resp = http_session().get(rec.download_url, timeout=90, stream=True)  # type: ignore[arg-type]
        resp.raise_for_status()
        chunks = bytearray()
        for chunk in resp.iter_content(_STREAM_CHUNK):
            chunks.extend(chunk)
        return bytes(chunks)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} available={self.available}>"
