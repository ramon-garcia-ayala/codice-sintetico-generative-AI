# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

It lives at `.claude/CLAUDE.md`, not the repo root — both are supported project-instruction locations
and load identically. `.claude/` is versioned so these instructions travel with the repo; only Claude
Code's per-machine runtime state under `.claude/` is gitignored. Don't "fix" this by moving the file
to the root or by adding an `@` import shim — nothing is broken.

## Project

This repo builds the training dataset and infrastructure for `codice_geo`, a SDXL LoRA trained on
geological strata and plastiglomerates for *Códice Sintético* (Fondo Creativo 2026). The LoRA is later
used to "fossilize" photogrammetric geometry of an urban fragment (Mercado San Juan de Dios, Guadalajara)
via ComfyUI ControlNet, producing morphology that gets 3D-printed in clay.

Three independent parts:

- **`scraper/`** — a Python package (`codice_scraper`) that audits, rescues, scrapes, curates, and
  exports the image dataset.
- **`runpod/`** — shell scripts + `kohya_ss` config to train the LoRA on a rented GPU pod.
- **`LoRA_train_workflows/` / `Image_workflows/`** — pre-existing Jupyter notebooks (SDXL/FLUX training
  and generation) from an unrelated IAAC course; kept for reference, not part of the `codice_geo` pipeline.
  `runpod/` uses `kohya_ss` instead of these notebooks — see "Why kohya, not the notebooks" below.

## Commands

**Fresh machine: follow `INIT.md` at the repo root.** It is the verified setup runbook (venv,
`pip install -e ".[dev]"`, and the three checks whose expected values are exact:
`145 passed`, 511 active records, 14 subcommands). It also documents what a clone
does *not* contain and how to get the images back — read it before assuming a
missing-image symptom is a bug.

All scraper commands run from `scraper/`:

```bash
pip install -e .                 # install codice_scraper (pillow, numpy, requests, pydantic, pyyaml, tqdm)
python -m pytest -q              # run all tests
python -m pytest tests/test_filters.py -q          # single file
python -m pytest tests/test_filters.py::test_name  # single test
```

CLI (installed as `codice-scraper`, or `python -m codice_scraper` from `scraper/`):

```bash
codice-scraper ingest                    # copy the Drive dataset into dataset/_incoming, seed manifest
codice-scraper audit --path dataset/_incoming --write-manifest
codice-scraper recover                   # recover provenance by pHash match against Wikimedia Commons
codice-scraper fetch [--klass X] [--source Y] [--dry-run]   # discover + download from new sources
codice-scraper classify [--overwrite]    # assign class from recovered Commons categories
codice-scraper caption [--overwrite]     # build kohya captions
codice-scraper reject <filenames...> [--reason ...]   # manual override after visual review
codice-scraper restore <filenames...>    # undo a manual reject (not an automatic filter reject)
codice-scraper refilter --min-short-side N [--klass X]   # re-apply resolution threshold on stored metrics, no re-download
codice-scraper relicense                 # re-apply the license policy on stored metadata, no re-download
codice-scraper synthesize --count 100    # SDXL + IP-Adapter variations, needs a local CUDA GPU + pip install -e .[synth]
codice-scraper sheet [--klass X] [--only-rejected]       # HTML contact sheet for visual review
codice-scraper export                    # write the kohya training tree
codice-scraper report [--attributions]   # manifest summary, optionally write ATTRIBUTIONS.md
```

Every command is idempotent — re-running never re-downloads or duplicates manifest entries. Run
`codice-scraper --help` for the full flag list; the module docstring in `__main__.py` documents the
intended command order end to end.

## Architecture

### The manifest is the source of truth

`dataset/manifest.jsonl` (one `ImageRecord` per line, keyed by filename) is the only state that
persists between commands. Every stage — audit, recover, fetch, classify, caption, export — reads
and rewrites it. Nothing is ever deleted from disk by the pipeline; images that shouldn't train are
marked `rejected=True` with one or more `RejectReason`s and stay in `dataset/_incoming/`, filterable
back in via `restore` or `refilter`. This mark-don't-delete design is what makes every command
idempotent and every filtering decision reversible and auditable.

### Sources discover, the pipeline downloads

`sources/base.py` defines `Source.search(query, klass, limit) -> Iterator[ImageRecord]`, which must
return candidates **without downloading image bytes** — this is what makes `fetch --dry-run` report
volume and licenses before spending bandwidth. `pipeline.fetch()` owns dedup (against the manifest's
known URLs/pHashes), license policy (`licenses.py`, applied once centrally — not per-source, see
below), the actual download, and measurement (`filters.measure`).

Download itself is *not* always a plain GET: `Source.download(rec) -> bytes` is a second, optional
hook (default: GET `rec.download_url`) that a source overrides when "one URL = one file" doesn't
hold. `EuropePMCSource` overrides it because Europe PMC's per-figure rendering endpoint
(`europepmc.org/articles/.../bin/...`) actively resets the connection for non-browser clients — the
only reliable path is the `supplementaryFiles` ZIP on `ebi.ac.uk`, fetched once per article (cached)
and split into per-figure `ImageRecord`s via `zip_url::member_name` encoded into `download_url`.

Adding a source means implementing `search()` (and `download()` only if needed) in
`sources/<name>.py` and registering it in `sources/__init__.py`. Nothing else changes.

### License policy: one definition, three enforcement points

`licenses.py` is the only place that *decides* whether a license string is trainable. It excludes NC
(the work is publicly exhibited and could see commercial reach) and ND (the whole pipeline transforms
the images). **Denied clauses must be checked before allowed markers**: `"cc by" in "cc by-nc-nd"` is
true, so checking "contains cc by" first silently admits NC/ND licenses. Never re-implement this
check inside a source — a source-local copy has already drifted out of sync once.

It is *enforced* at three points, and all three are load-bearing:

- `pipeline.fetch()` — at discovery, so unusable material is never downloaded.
- `recover._apply_metadata()` via `apply_license_policy()` — the legacy Drive images arrive through
  `ingest`, never through `fetch`, and get their license here. Without this the whole
  `ingest → recover → export` path bypassed the policy and published unvetted images in
  ATTRIBUTIONS.md as verified.
- `export_kohya()` — the boundary where an image stops being a candidate and becomes published
  material, so no future path into the manifest can leak into `dataset/train/`.

Rejected licenses are marked `license_denied` (visible in `report` and the contact sheet), not
dropped. `codice-scraper relicense` re-applies the policy to an existing manifest without hitting
the network — use it after changing `licenses.py`.

### Who may revert a rejection

`restore` reverts only `RejectReason.MANUAL`. This is functional, not cosmetic: hand-reverting an
automatic verdict that the next `audit` would re-apply is a loop with no exit. So automatic verdicts
must never be stamped `MANUAL` — `classify`'s catalogue-noise rejection uses `OUT_OF_SCOPE`
precisely so a batch `restore` cannot silently readmit it.

Symmetrically, `reject` always records `MANUAL`, even on an image a filter already rejected.
Skipping that case left the human decision unrecorded, and a later `refilter` (which lifts the
automatic reason) would return it to training with no trace.

`apply_filters` recomputes the pixel-derived reasons in `RECOMPUTED_REJECT_REASONS` from scratch on
every pass and preserves everything else. Without that, lowering a threshold is a silent no-op and
`refilter` rescues get undone by the next `audit`.

### Four training classes, encoded as kohya repeat-count folders

`models.ImageClass` + `CLASS_REPEATS` define the dataset's four buckets and their kohya repeat
prefix (`10_01_real_estratos`, `15_02_real_plastiglomerado`, `8_03_proxy_materiales`,
`5_04_synth_plastiglomerado`). Repeats — not file duplication — are how wildly uneven class sizes
(hundreds of strata vs. a handful of real plastiglomerate photos) get balanced per training epoch.
`export/kohya.py` is the only place that reads `CLASS_REPEATS` and writes the `dataset/train/`
tree; it also emits `captions_SDXL.csv` for compatibility with
`LoRA_train_workflows/01_SDXL/02_SDXL_LoRA_Captions_Check.ipynb`.

`classify.py` assigns a class from the categories `recover`/`fetch` attached to a record.
`load_group_index()` indexes **both** `wikimedia_categories` and `search_terms` from
`config/queries.yaml` — a record discovered via free-text search (Europe PMC, or Wikimedia's
text-search path) carries the search term itself in `categories` (`record_from_page` /
`recover._apply_metadata` append it), so skipping `search_terms` in the index makes `classify
--overwrite` blank out any record whose class came from a free-text match instead of a curated
Commons category. This happened for real: it silently reclassified one of only two real
plastiglomerate images to `UNCLASSIFIED` and wiped its caption. `classify()` deliberately leaves a
record `UNCLASSIFIED` rather than guess when there's no category *or* search-term signal — guessing
under an already-unknown license is compounding uncertainty no one can audit later.

**`--overwrite` re-derives class for every record it touches**, including ones a `fetch --klass X`
call already tagged correctly. It's the right tool for fixing a genuine mis-mapping (the
`legacy_noise` → `OUT_OF_SCOPE` case), but re-running it is not a no-op safety net — verify the
resulting class counts (`codice-scraper report`) match expectations before moving on.

### Filtering: pixel heuristics catch what they catch, nothing more

`filters.py` implements resolution, aspect-ratio (panoramic bucket vs. hard reject), duplicate
(sha256 exact + pHash Hamming ≤ 5), sharpness (Laplacian variance), and studio/vitrine-background
detection — all in `hashing.py`'s pure-numpy pHash/Laplacian, no `imagehash` or OpenCV dependency.
The studio-background check works by looking for a uniform, extreme (near-black or near-white)
perimeter border; it does **not** catch a specimen photographed close enough to fill the frame (no
clean border to measure), nor a museum vitrine shot whose background is a busy wall/glass-reflection
rather than a uniform one. Both need the `needs_review` text-based flags in `classify.py`
(`_flag_studio_language`: "specimen" in the description, "museum"/"博物館"/etc. in the filename or
title) plus a human decision in the contact sheet, not a stricter pixel rule. When adjusting a
threshold, `codice-scraper audit --path dataset/_incoming` should reproduce the same counts
documented in `scraper/tests/` (`test_filters.py`) — those numbers are the regression test.

### Synthetic plastiglomerate: `synthesize.py`

Real plastiglomerate photography is the scarce class — only 2 verified real images survived curation
(see `scraper/README.md`). `codice-scraper synthesize` generates variations locally with SDXL +
IP-Adapter (`h94/IP-Adapter`), reimplementing `Image_workflows/04_SDXL_IP.ipynb` outside Colab (that
notebook imports `google.colab.drive`/`userdata` directly, so it can't run as-is anywhere else) —
needs a local CUDA GPU and `pip install -e .[synth]` (torch/diffusers/accelerate are not base
dependencies, imported lazily only inside this module).

**IP-Adapter scale is 0.35, not the notebook's 0.6.** At 0.6, conditioned on this project's 2 real
references, the model reproduced composition *and context* almost verbatim — including one
reference's museum vitrine (glass, wall, pedestal) in full — rather than transferring material style.
At 0.35 the text prompt (which always specifies beach/dune/volcanic contexts, never museum) regains
real influence over composition. The vitrine reference is also used pre-cropped
(`scraper/synth_refs/wm_plastiglomerate_museon_crop.jpg`) to reduce how much museum context is even
available to condition on. Verify any new reference image or scale change by generating a handful and
inspecting them before committing to a full batch — this is exactly the kind of failure a total-count
check can't catch, only looking at the pixels can.

Each synthetic record's `license` field reads `"CC BY-SA 4.0 (síntesis derivada, no descargada de
Commons)"` rather than a plain Commons license string — it's a derivative work of a CC BY-SA image,
not a Commons download, and `ATTRIBUTIONS.md` needs that distinction to stay honest about provenance.

### Why kohya, not the notebooks

`train_codice_geo.toml` targets `kohya_ss`, not the diffusers-based
`LoRA_train_workflows/01_SDXL/03_SDXL_LoRA_Train.ipynb` already in this repo. kohya emits
`.safetensors` ComfyUI loads directly (diffusers emits PEFT format, requiring the conversion step
`04_SDXL_LoRA_Convert_Use.ipynb` exists for) and supports the per-folder repeat mechanism the four
classes rely on. See `runpod/README.md` for the full training-parameter rationale (bucketing, no
flip-aug, VAE fp16-fix, etc.) and `scraper/README.md` for the dataset-curation rationale.

### Why dataset images are never versioned

`.gitignore` excludes `dataset/**` except `manifest.jsonl` and `ATTRIBUTIONS.md`, so a clone is
~9 MB against 4.4 GB on disk (2.4 GB `_incoming/`, 2.1 GB `train/` — which is a duplicate copy of
the selected 511). Three reasons, and the first is not about size:

- **`_incoming/` holds 20 images whose license is `UNKNOWN`**, on purpose: the mark-don't-delete
  design keeps rejected material next to accepted material in the same directory. Committing
  `dataset/**` wholesale publishes unvetted images in a publicly exhibited project, bypassing the
  license policy that the three enforcement points above exist to guarantee. Git LFS doesn't help —
  the problem is publication, not size.
- **Git history is permanent.** Removing an image later needs a history rewrite and a force-push.
  This repo already paid that cost once, to drop 217 MB of reference notebooks.
- **The manifest is the part that can't be rebuilt.** Losing the original filenames is what
  destroyed provenance in the first scrape and forced the pHash recovery. Wikimedia pixels can
  always be re-downloaded; `download_url` + `sha256` + license + author + caption cannot.

Image backups belong in the shared Drive, not in git. **The 100 synthetics are the only
irrecoverable files** — no `download_url`, and regenerating them needs a GPU and isn't
bit-identical across hardware.

Related trap: **`fetch` does not rehydrate a manifest whose files are missing.** Dedup compares the
URL against the manifest without checking the disk (`pipeline.py:203`), so on a fresh clone it
reports everything as already known and downloads nothing (measured: 141 discovered / 141 known,
zero files present). Re-downloading means walking the manifest's `download_url`s directly.

## External dependencies with known constraints

- **The Google Drive at `paths.DRIVE_ROOT`** (env override: `CODICE_DRIVE`) is read-only from this
  codebase's perspective — it's the shared project Drive, synced to the rest of the team. `ingest`
  copies from it into `dataset/_incoming/`; no command ever writes back into it.
- **USGS and BGS GeoScenic are not usable as sources** (checked 2026-08-10: 404/503/unreachable) —
  don't add clients for them without re-verifying first.
- **Europe PMC's `europepmc.org` figure-rendering endpoint blocks non-browser clients**; only the
  `ebi.ac.uk` `supplementaryFiles` ZIP endpoint works. NCBI PMC mirrors return 403.
- **`Category:Plastiglomerate` on Wikimedia Commons is empty.** Real plastiglomerate imagery is
  scarce and lives in scientific literature, not image banks — and even there, openly-licensed
  figures are usually web-embedded resolution (600–850px), well under SDXL's 1024px target.
