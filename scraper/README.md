# codice-scraper

Construcción del dataset geológico para el LoRA `codice_geo` de *Códice
Sintético* — paso 4 del brief: 300–500 imágenes de estratos sedimentarios y
plastiglomerados, listas para `kohya_ss`.

## Qué problema resuelve

En `02_DATASET` ya había 282 imágenes de un scrape previo de Wikimedia. La
auditoría encontró tres problemas que impedían entrenar:

| | |
|---|---|
| **0 captions** | kohya entrena con caption vacío y no avisa |
| **Sin clases** | todo plano, no se distingue estrato de plastiglomerado |
| **Sin procedencia** | el renombrado a `wikimedia_###` borró título, autor y licencia |

Más 33 imágenes (11%) que son fotografía de vitrina de museo —minerales de
colección sobre fondo negro, alguna con la cartela impresa— y que enseñarían al
modelo a pintar fondos planos y letras.

El paquete audita, rescata la procedencia, clasifica, escribe captions y
exporta el árbol de entrenamiento.

## Instalación

```bash
cd scraper
pip install -e .
```

Sólo necesita `pillow`, `numpy`, `requests`, `pydantic`, `pyyaml` y `tqdm`.
El pHash y la varianza del laplaciano están implementados en numpy puro
(`hashing.py`) para no arrastrar `imagehash` ni OpenCV.

## Flujo completo

```bash
codice-scraper ingest                    # copia el Drive a dataset/_incoming
codice-scraper audit --path dataset/_incoming --write-manifest
codice-scraper recover                   # procedencia por pHash contra Commons
codice-scraper fetch                     # amplía con las fuentes nuevas
codice-scraper classify                  # clase según categorías de Commons
codice-scraper caption                   # captions con el trigger codice_geo
codice-scraper sheet                     # hoja de contacto para revisar
codice-scraper export                    # árbol N_clase/ para kohya
codice-scraper report --attributions     # estado + ATTRIBUTIONS.md
```

Todo es idempotente: repetir un comando no vuelve a descargar ni duplica
entradas del manifest.

**El Drive de Carlos es de sólo lectura.** `ingest` copia a
`dataset/_incoming/` y todo lo demás trabaja ahí. Una escritura in-place se
sincronizaría a todo el equipo.

## `recover` — rescatar la procedencia

Los nombres originales se perdieron, pero las imágenes siguen en Commons. El
módulo barre las mismas categorías, calcula el pHash de cada miniatura remota y
empareja contra las locales con distancia de Hamming ≤ 5. Un match devuelve
título, autor, licencia, descripción y categorías.

Se descartó la vía obvia —buscar por SHA-1 exacto, que Commons soporta— porque
se comprobó que el scrape original recomprimió los archivos: 0 de 6 dieron
match. El pHash es robusto a esa recompresión; el hash criptográfico no.

Lo que no se empareja queda como `license: UNKNOWN` en el manifest y aparece en
`ATTRIBUTIONS.md` bajo "sin procedencia identificada". Marcado, no escondido.

## Las cuatro clases

| Carpeta | Repeats | Contenido |
|---|---:|---|
| `01_real_estratos` | 10 | Estratos, roca sedimentaria, afloramientos |
| `02_real_plastiglomerado` | 15 | Plastiglomerado real documentado |
| `03_proxy_materiales` | 8 | Escoria, pyroplastics, brechas con inclusiones |
| `04_synth_plastiglomerado` | 5 | Variaciones SDXL sobre las reales |

El prefijo numérico son las repeticiones por época de kohya. Es lo que equilibra
clases de tamaños muy distintos sin duplicar archivos: hay cientos de estratos y
sólo decenas de plastiglomerados reales.

Las sintéticas llevan el token `codice_synth` además del trigger, para poder
medir su efecto y retirarlas del prompt sin reentrenar.

## Fuentes

| Fuente | Clave | Notas |
|---|---|---|
| `wikimedia` | no | Principal para estratos. Por categoría y por texto |
| `europepmc` | no | **Clave para plastiglomerado** |
| `flickr` | sí | Campo y macro. Sin key queda deshabilitada |

### Dos fuentes del brief que no están disponibles

El brief nombraba cuatro fuentes. Se comprobaron todas (2026-08-10) y **dos no
son utilizables**:

- **USGS** — `library.usgs.gov/photo/api` responde 404, `libs.er.usgs.gov` no
  resuelve, y ScienceBase devuelve 0 resultados para fotografía de estratos.
  Las categorías USGS que sí hay en Commons son casi todas fotografía aérea de
  Virginia, inservible para textura de roca.
- **BGS GeoScenic** — `geoscenic.bgs.ac.uk` responde **503** y la ruta
  alternativa en `bgs.ac.uk` da 404. Puede ser caída temporal; conviene
  reintentar antes de descartarla del todo.

No se escribió código contra ninguna de las dos: implementar un cliente contra
un endpoint que devuelve 404 sólo produce una fuente que falla en silencio.

La pérdida se compensa con Wikimedia, que cubre estratos de sobra: sólo las
seis categorías principales suman ~1.500 archivos directos frente a las 300–400
que pide el brief.

Dos cosas que se midieron y conviene saber:

- **`Category:Plastiglomerate` en Commons devuelve 0 archivos.** El corpus
  público de plastiglomerado no está en bancos de imágenes: está en literatura
  científica. Europe PMC devuelve ~190 candidatas para los términos del
  dominio (*pyroplastic*, *anthropocene plastic sediment*, Kamilo Beach).
- **`Marine debris` y `Plastic pollution` se probaron y se quitaron.** Aportaban
  ~120 fotos de botellas y redes en la playa: plástico suelto sin matriz
  mineral. Al ser plastiglomerado la clase escasa, la habrían contaminado más
  que a ninguna otra.

Las figuras de Europe PMC suelen ser compuestas (paneles a/b/c con barras de
escala). Se marcan en `needs_review` y **no se recortan automáticamente**: eso
se decide con la imagen delante en la hoja de contacto.

## Licencias

`licenses.py` centraliza la política y el pipeline la aplica a todas las
fuentes por igual. Se excluyen **NC** (el proyecto se expone en una institución
y puede tener difusión comercial) y **ND** (todo el pipeline es transformación,
que es justo lo que ND prohíbe).

El orden de comprobación importa: `"cc by" in "cc by-nc-nd"` es verdadero, así
que hay que excluir las cláusulas prohibidas **antes** de buscar las permitidas.
Un filtro por inclusión ingenuo deja pasar precisamente lo que quiere bloquear
— pasó, y por eso existe `test_licenses.py`.

## Filtrado

Umbrales en `FilterConfig`, todos configurables. Nada se borra: se marca
`rejected` con su razón, y la hoja de contacto permite revisarlo.

- **Fondo de estudio** — marco perimetral uniforme y extremo, o imagen
  mayoritariamente negra o blanca. Detectó las 33 del dataset heredado.
- **Resolución** — lado corto ≥ 1024 px. SDXL entrena a 1024.
- **Aspecto** — ≥ 2.0 va al bucket panorámico (la sección madre del brief es
  3:1); > 3.5 se descarta.
- **Duplicados** — `sha256` para idénticos, pHash Hamming ≤ 5 para reescalados.
- **Nitidez** — varianza del laplaciano.

Nota: más de la mitad del dataset tiene saturación baja. Es esperable en caliza
y roca gris, así que **no se trata como defecto**; se nombra en el caption
(`desaturated grey tones`) para que el modelo la trate como atributo separable
en vez de fijarla dentro del concepto.

## Tests

```bash
python -m pytest -q
```

Cubren pHash y su estabilidad ante reescalado y recompresión, los umbrales de
descarte contra los casos reales del dataset, idempotencia del manifest, la
política de licencias y el export a kohya.

El test de regresión de verdad es la auditoría sobre las 282 originales, que
debe seguir dando **33 fondo de estudio, 16 bajo 1024 px y 10 panorámicas**.
