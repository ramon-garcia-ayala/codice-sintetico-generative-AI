# Códice Sintético — `codice_geo`

Dataset, entrenamiento e infraestructura del LoRA **`codice_geo`**: un modelo SDXL
entrenado sobre estratos sedimentarios y plastiglomerados para *Códice Sintético*
(Fondo Creativo 2026).

El LoRA no es la obra: es la herramienta. Sirve para **fosilizar** la geometría
fotogramétrica de un fragmento urbano —el Mercado San Juan de Dios, Guadalajara—
mediante ComfyUI + ControlNet, produciendo una morfología que después se
**imprime en 3D en arcilla**.

---

## La cadena completa

| Paso | Qué pasa | Dónde vive |
|---:|---|---|
| 1–3 | Escaneo fotogramétrico del fragmento urbano y simulaciones en Blender | Drive del proyecto (`01_ESCANEO`) |
| **4** | **Construcción y curación del dataset geológico** | **`scraper/`** |
| **5** | **Entrenamiento del LoRA en una GPU rentada** | **`runpod/`** |
| 6–7 | Mapas de *depth* y *normal* desde los `.blend` (Z-pass, Normal-pass) | Drive (`01_ESCANEO/SIMULACIONES`) |
| 8 | Generación en ComfyUI: SDXL + `codice_geo` + ControlNet | Fuera de este repo |
| 9 | Impresión 3D en arcilla | Fuera de este repo |

Este repositorio cubre los pasos **4 y 5**. Los pesos entrenados se suben al
Drive (`03_LORA`) y de ahí los toma ComfyUI.

---

## Qué hay en el repo

```
scraper/     Paquete Python `codice_scraper` — audita, rescata procedencia,
             descarga, filtra, clasifica, captiona y exporta el dataset.
runpod/      Scripts de shell + config `kohya_ss` para entrenar en un pod.
dataset/     Sólo `manifest.jsonl` y `ATTRIBUTIONS.md`. Las imágenes NO se versionan.
INIT.md      Runbook verificado de puesta en marcha en una máquina nueva.
.claude/     Instrucciones de proyecto para Claude Code (`CLAUDE.md`).
```

Los notebooks `LoRA_train_workflows/` e `Image_workflows/` son material de
referencia de un curso ajeno al proyecto: viven en disco pero están fuera del
repo a propósito (~217 MB de salidas embebidas, y el pipeline entrena con
`kohya_ss`, no con ellos).

---

## Arranque rápido

**En una máquina nueva, seguir [`INIT.md`](INIT.md).** Es el runbook probado
sobre un clon limpio, con los valores exactos que debe dar cada verificación.

Resumen:

```bash
cd scraper
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"   # Windows
# ./.venv/bin/python -m pip install -e '.[dev]'          # macOS / Linux

./.venv/Scripts/python.exe -m pytest -q                  # → 146 passed
./.venv/Scripts/python.exe -m codice_scraper report      # → 1112 entradas, 471 activas
```

Las tres verificaciones pasan **sin red, sin GPU, sin el Drive montado y sin una
sola imagen en disco**: sólo leen el manifest versionado.

Flujo del pipeline de datos, en orden:

```bash
codice-scraper ingest                 # copia el Drive a dataset/_incoming
codice-scraper audit --path dataset/_incoming --write-manifest
codice-scraper recover                # procedencia por pHash contra Wikimedia Commons
codice-scraper fetch                  # amplía con fuentes nuevas
codice-scraper classify               # asigna clase
codice-scraper caption                # captions con el trigger codice_geo
codice-scraper sheet                  # hoja de contacto para revisión humana
codice-scraper export                 # árbol N_clase/ que consume kohya
codice-scraper report --attributions  # estado + ATTRIBUTIONS.md
```

Los 14 subcomandos son idempotentes: repetir uno no vuelve a descargar ni
duplica entradas. Detalle completo en [`scraper/README.md`](scraper/README.md).

---

## Estado del dataset

| | |
|---|---:|
| Entradas en el manifest | 1112 |
| **Aprobadas para entrenar** | **471** |
| Descartadas (marcadas, no borradas) | 641 |
| Con procedencia verificada | 471 / 471 |
| Con caption | 471 / 471 |

Las cuatro clases y sus repeticiones por época en kohya:

| Carpeta | Repeats | Imágenes | Contenido |
|---|---:|---:|---|
| `01_real_estratos` | 10 | 216 | Estratos, roca sedimentaria, afloramientos |
| `02_real_plastiglomerado` | 15 | 2 | Plastiglomerado real documentado |
| `03_proxy_materiales` | 8 | 153 | Escoria, *pyroplastics*, brechas con inclusiones |
| `04_synth_plastiglomerado` | 5 | 100 | Variaciones SDXL + IP-Adapter sobre las 2 reales |

Los *repeats* —no la duplicación de archivos— son lo que equilibra clases de
tamaños radicalmente distintos. Sólo **2** imágenes de plastiglomerado real
sobrevivieron a la curación: `Category:Plastiglomerate` en Commons está vacía y
el corpus público vive en literatura científica, casi siempre a resolución
web (600–850 px), por debajo del objetivo de 1024 px de SDXL. De ahí las clases
*proxy* y *synth*.

---

## Cuatro decisiones que explican el diseño

**El manifest es la única fuente de verdad.** `dataset/manifest.jsonl` (un
registro por imagen) es el único estado que persiste entre comandos. Todas las
etapas lo leen y lo reescriben.

**Marcar, no borrar.** El pipeline nunca elimina un archivo del disco. Lo que no
debe entrenar queda con `rejected=True` y sus razones, en la misma carpeta que lo
aprobado, reversible vía `restore` o `refilter`. Eso es lo que hace cada comando
idempotente y cada decisión auditable.

**La política de licencias se define una vez y se aplica en tres.** `licenses.py`
excluye NC (la obra se exhibe públicamente) y ND (todo el pipeline es
transformación), y se hace valer al descubrir (`fetch`), al recuperar metadatos
(`recover`) y al publicar (`export`). Ninguna fuente reimplementa esa
comprobación: una copia local ya se desincronizó una vez.

**Las imágenes nunca se versionan.** El clon pesa ~9 MB contra 4.4 GB en disco, y
la razón principal no es el tamaño: `_incoming/` conserva a propósito 20 imágenes
con licencia `UNKNOWN`, y subir la carpeta en bloque publicaría material sin
verificar en un proyecto con exhibición pública. Git LFS no resuelve eso — el
problema es la publicación, no los bytes. Lo irreemplazable es el manifest: los
píxeles de Wikimedia se vuelven a bajar, la procedencia no se reconstruye.
Perderla fue el daño original que hubo que rescatar por pHash.

---

## Entrenamiento

`runpod/` entrena con `kohya_ss` sobre una **RTX 4090** (24 GB): unos 3.500 pasos
por época, **40–60 minutos** por corrida. kohya —y no los notebooks de
diffusers— porque emite directamente el `.safetensors` que ComfyUI carga sin
conversión, y porque soporta el mecanismo de *repeats* por carpeta del que
dependen las cuatro clases.

```bash
git clone <este-repo> /workspace/codice && cd /workspace/codice/runpod
bash bootstrap.sh      # kohya + modelos base
bash sync.sh setup && bash sync.sh pull
bash train.sh --smoke  # 10 pasos: valida el árbol antes de gastar GPU
bash train.sh
bash sync.sh push      # pesos a 03_LORA
```

El *smoke test* no es opcional: si a una imagen le falta su `.txt`, kohya entrena
con caption vacío **sin avisar** y el fallo sólo se descubre al generar.
Justificación de cada parámetro en [`runpod/README.md`](runpod/README.md).

---

## Documentación

| Documento | Para qué |
|---|---|
| [`INIT.md`](INIT.md) | Puesta en marcha en una máquina nueva, con criterios exactos de verificación |
| [`scraper/README.md`](scraper/README.md) | Curación del dataset: fuentes, filtros, licencias, síntesis |
| [`runpod/README.md`](runpod/README.md) | Entrenamiento: GPU, volumen, parámetros y su porqué |
| [`dataset/ATTRIBUTIONS.md`](dataset/ATTRIBUTIONS.md) | Autoría y licencia de cada imagen usada |
| [`.claude/CLAUDE.md`](.claude/CLAUDE.md) | Arquitectura e invariantes, para trabajar sobre el código |

---

## Atribución

El dataset se construye con material de Wikimedia Commons, Europe PMC y Flickr
bajo licencias CC compatibles con obra derivada y uso comercial. La autoría
completa, imagen por imagen, está en
[`dataset/ATTRIBUTIONS.md`](dataset/ATTRIBUTIONS.md); las imágenes cuya
procedencia no pudo verificarse aparecen ahí marcadas como tales, no ocultas.
