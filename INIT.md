# INIT — puesta en marcha en una máquina nueva

Runbook para Claude Code. Al abrir este repo en otra computadora basta con pedir
*«ejecuta INIT.md»*: los pasos de abajo dejan el proyecto corriendo y verificado.

Todos los comandos y números de este archivo se probaron sobre un clon limpio en
una ruta distinta con un venv virgen (Python 3.13.12, Windows). No son estimados.

---

## 0. Qué NO viene en el repo — leer antes de empezar

El clon pesa **~9 MB**. Lo que falta, falta a propósito:

| Ausente | Tamaño | Por qué |
|---|---:|---|
| Las imágenes del dataset | 4.4 GB | Decisión explícita: ver *"Why dataset images are never versioned"* en `CLAUDE.md` |
| `scraper/synth_refs/` | 8 MB | Copias/recortes de imágenes que ya viven en `dataset/`; reconstrucción documentada en `scraper/README.md` |
| `scraper/config/secrets.yaml` | — | Credenciales de Flickr |

**Lo que sí viene, y es lo importante:** `dataset/manifest.jsonl` (1.5 MB) con la
procedencia completa de las 511 imágenes aprobadas — URL, `sha256`, `phash`,
licencia, autor, categorías y caption. Los píxeles de Wikimedia se vuelven a
bajar; los metadatos no se reconstruyen (perderlos fue el daño original que hubo
que rescatar por pHash). Por eso el manifest se versiona y las imágenes no.

Consecuencia práctica: **tras clonar, el pipeline arranca sin imágenes en disco.**
`codice-scraper export` responderá `Nada que exportar`. Eso es correcto, no un
fallo. Ver el paso 4.

---

## 1. Requisitos

- **Python ≥ 3.11** (`requires-python` en `scraper/pyproject.toml`; verificado con 3.13).
- **git**.

Nada más para el pipeline base: pHash y varianza del Laplaciano están en numpy
puro, sin `imagehash` ni OpenCV. GPU CUDA sólo hace falta para
`codice-scraper synthesize` (paso 5).

---

## 2. Entorno virtual e instalación

El venv vive en `scraper/.venv` (ya está en `.gitignore`).

**PowerShell (Windows):**

```powershell
cd scraper
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

**bash / zsh:**

```bash
cd scraper
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e '.[dev]'   # Windows
# ./.venv/bin/python -m pip install -e '.[dev]'         # macOS / Linux
```

Notas de las dos que muerden en la práctica:

- Las comillas alrededor de `".[dev]"` no son opcionales: sin ellas los
  corchetes se interpretan como glob y la instalación falla o se queda sin pytest.
- Se invoca `.venv\Scripts\python.exe` **directamente en lugar de activar** el
  entorno. Es deliberado: `Activate.ps1` puede quedar bloqueado por la
  ExecutionPolicy de PowerShell, y llamar al intérprete por ruta funciona
  siempre. Si prefieres activar, `.\.venv\Scripts\Activate.ps1` y luego `python`
  a secas es equivalente.

El extra `[dev]` trae `pytest`. Las dependencias base son 6:
`pillow`, `numpy`, `requests`, `pydantic`, `pyyaml`, `tqdm`.

---

## 3. Verificación — criterios exactos

Corre las tres y compara contra los valores esperados. **Si algún número no
coincide, detente y repórtalo; no intentes "arreglarlo" re-corriendo etapas del
pipeline** — `classify --overwrite` y compañía pueden empeorar el estado
(precedente documentado en `CLAUDE.md`).

```bash
./.venv/Scripts/python.exe -m pytest -q
```
→ **`145 passed`** (~6 s).

```bash
./.venv/Scripts/python.exe -m codice_scraper report
```
→ debe reproducir exactamente:

```
Entradas         1112        Activas   511        Descartadas   601
01_real_estratos  216 | 02_real_plastiglomerado  2
03_proxy_materiales 193 | 04_synth_plastiglomerado 100
Con procedencia   511/511  (100%)
Con caption       511/511
```

```bash
./.venv/Scripts/python.exe -m codice_scraper --help
```
→ 14 subcomandos: `audit ingest recover fetch refilter relicense synthesize
classify caption sheet export reject restore report`.

Estos tres pasan **sin red, sin GPU, sin el Drive montado y sin imágenes en
disco**: sólo leen el manifest versionado.

---

## 4. Recuperar las imágenes (sólo si hay que re-entrenar o re-exportar)

Para leer el manifest, generar reportes o tocar el código, esto **no** hace falta.

**Vía recomendada — el Drive del equipo.** Es el almacenamiento compartido y ya
contiene `02_DATASET`:

```bash
export CODICE_DRIVE="/ruta/al/Drive/CODICE SIN"     # bash
$env:CODICE_DRIVE = "G:\...\CODICE SIN"             # PowerShell
./.venv/Scripts/python.exe -m codice_scraper ingest
```

`ingest` recupera las **282 heredadas**. Si el equipo respaldó el dataset
completo en el Drive, copia esos archivos a `dataset/_incoming/` y corre
`codice-scraper export` directamente — el manifest ya tiene clase y caption de
las 511, no hay que re-clasificar nada.

`DRIVE_ROOT` es de **sólo lectura** para este paquete: `ingest` copia hacia
`dataset/_incoming/` y ningún comando escribe de vuelta. No cambies eso.

**Trampa verificada: `codice-scraper fetch` NO rehidrata.** La deduplicación
compara la URL contra el manifest sin comprobar si el archivo existe en disco
(`pipeline.py:203`). Sobre un clon limpio el resultado real fue
`Descubiertas 141 / Ya conocidas 141` con cero archivos en disco y cero
descargas. Para re-bajar las 411 de Wikimedia hay que ir contra los
`download_url` del manifest (y validar con el `sha256` guardado), no vía `fetch`.

**Las 100 sintéticas no se recuperan de ninguna de las dos vías.** No están en
git, no tienen `download_url`, y regenerarlas exige GPU y no sale bit-idéntico
entre hardware distinto. Si no hay copia en el Drive, están sólo en la máquina
donde se generaron.

---

## 5. Extras opcionales

**Síntesis con SDXL + IP-Adapter** (`codice-scraper synthesize`) — necesita GPU CUDA:

```bash
./.venv/Scripts/python.exe -m pip install -e '.[synth]'
```

El extra fija `tokenizers<0.23` a propósito: sin ese techo entra `0.23.0rc0` y
rompe el tokenizer CLIP de SDXL con
`RobertaProcessing.__new__() got an unexpected keyword argument 'cls'`. En
Windows, si la descarga del modelo falla con `OSError [WinError 1]` por
symlinks, exporta `HF_HUB_DISABLE_SYMLINKS=1`. Requiere además reconstruir
`scraper/synth_refs/` (ver `scraper/README.md`).

**Flickr** — crea `scraper/config/secrets.yaml` (gitignored):

```yaml
flickr:
  api_key: "..."
```

Sin ese archivo la fuente se omite con un aviso; no rompe nada.

**Entrenamiento** — los scripts de `runpod/` corren **en el pod**, no en local:
`bootstrap.sh` → `sync.sh setup` (login interactivo de Google Drive) →
`sync.sh pull` → `train.sh --smoke` → `train.sh` → `sync.sh push`. Detalle de
parámetros en `runpod/README.md`.

---

## 6. Invariantes que no se tocan

1. **Nunca escribir en el Drive.** Está sincronizado con todo el equipo.
2. **Nunca versionar imágenes.** Ni con Git LFS: el `_incoming/` conserva 20
   imágenes con licencia `UNKNOWN` (diseño *marcar-no-borrar*), y subir la
   carpeta en bloque publicaría material sin licencia verificada en un proyecto
   con exhibición pública. El historial de git es permanente.
3. **Nunca reimplementar la política de licencias** fuera de `licenses.py`, y
   las cláusulas denegadas se comprueban **antes** de los marcadores permitidos
   (`"cc by" in "cc by-nc-nd"` es `True`). Tras editar `licenses.py`, correr
   `codice-scraper relicense`.
4. Tras cualquier cambio: `pytest -q` debe seguir en **145 passed**.
