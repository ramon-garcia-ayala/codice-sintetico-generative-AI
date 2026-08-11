# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

`classify.py` assigns a class from the Wikimedia categories `recover`/`fetch` attached to a record
(`config/queries.yaml` maps category → class group); it deliberately leaves a record `UNCLASSIFIED`
rather than guess when there's no category signal, since guessing under an already-unknown license is
compounding uncertainty no one can audit later.

### Filtering: pixel heuristics catch what they catch, nothing more

`filters.py` implements resolution, aspect-ratio (panoramic bucket vs. hard reject), duplicate
(sha256 exact + pHash Hamming ≤ 5), sharpness (Laplacian variance), and studio/vitrine-background
detection — all in `hashing.py`'s pure-numpy pHash/Laplacian, no `imagehash` or OpenCV dependency.
The studio-background check works by looking for a uniform, extreme (near-black or near-white)
perimeter border; it does **not** catch a specimen photographed close enough to fill the frame (no
clean border to measure) — those need the `needs_review` text-based flag in `classify.py`
(descriptions containing "specimen") plus a human decision in the contact sheet, not a stricter pixel
rule. When adjusting a threshold, `codice-scraper audit --path dataset/_incoming` should reproduce
the same counts documented in `scraper/tests/` (`test_filters.py`) — those numbers are the
regression test.

### Why kohya, not the notebooks

`train_codice_geo.toml` targets `kohya_ss`, not the diffusers-based
`LoRA_train_workflows/01_SDXL/03_SDXL_LoRA_Train.ipynb` already in this repo. kohya emits
`.safetensors` ComfyUI loads directly (diffusers emits PEFT format, requiring the conversion step
`04_SDXL_LoRA_Convert_Use.ipynb` exists for) and supports the per-folder repeat mechanism the four
classes rely on. See `runpod/README.md` for the full training-parameter rationale (bucketing, no
flip-aug, VAE fp16-fix, etc.) and `scraper/README.md` for the dataset-curation rationale.

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
