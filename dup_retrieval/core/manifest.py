"""Stage A — text manifest extraction.

Read every ``cuts.*.jsonl.gz`` shard (no audio decode) and emit one parquet table:

    schema: (dataset, cut_id, recording_id, language, text, normalized_text,
             duration_secs, num_samples,
             tar_path, tar_offset, tar_size, audio_format)

The last four columns are the *random-access locator*: each cut's audio lives in
the paired ``recording.*.tar`` at byte ``tar_offset`` (length ``tar_size``), so
downstream stages ``seek``+``read`` one member instead of streaming the tar.

Composite primary key: (dataset, cut_id) — a ``cut_id`` may repeat across
datasets, but not within one. Runs under torchrun (SPMD over CPU ranks); each
rank takes shards round-robin and writes its own part.

Output: <output_dir>/manifest/part_{rank:04d}.parquet + _SUCCESS
"""

import argparse
import gzip
import json
import logging
import multiprocessing as mp
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SHAR_INDEX_FILENAME = "shar_index.json"
SUCCESS_MARKER = "_SUCCESS"

# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

# A subset of Unicode emoji + symbol ranges; we strip these unconditionally
# because they're informationally noisy for transcript dedup.
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"   # symbols & pictographs, transport & map
    "\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs extended-A
    "\U00002600-\U000026FF"   # misc symbols
    "\U00002700-\U000027BF"   # dingbats
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "\U0000200D"              # ZWJ
    "]+",
    flags=re.UNICODE,
)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[!-/:-@[-`{-~¡-¿]+")
# Collapse a run of the *same* punctuation char ("!!!" -> "!", "..." -> ".") so
# chained punctuation doesn't make otherwise-identical transcripts diverge.
_PUNCT_RUN_RE = re.compile(r"([!-/:-@[-`{-~¡-¿])\1+")
_ARABIC_DIACRITICS_RE = re.compile(r"[ً-ٰٟۖ-ۭ]")

CJK_LANGS = frozenset({"zh", "ja", "ko", "yue", "lzh"})
RTL_LANGS = frozenset({"ar", "he", "fa", "ur"})


def normalize_text(text: str, language: Optional[str] = None,
                   strip_punctuation: bool = False) -> str:
    """Language-aware text normalization for fuzzy dedup keys.

    Pipeline:
      1. NFKC normalize (compose, fold compat forms)
      2. casefold
      3. strip emoji + ZWJ
      4. language-specific:
         - Arabic/Hebrew/Farsi/Urdu: strip diacritics + alef variants
         - CJK: no further whitespace handling (n-grams will be char-level)
      5. collapse chained/repeated punctuation
      6. optional punctuation strip
      7. collapse whitespace
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = _EMOJI_RE.sub("", text)
    text = _PUNCT_RUN_RE.sub(r"\1", text)

    lang = (language or "").lower().split("-")[0] if language else ""

    if lang in RTL_LANGS:
        text = _ARABIC_DIACRITICS_RE.sub("", text)
        # Arabic alef variants -> bare alef
        text = text.translate(str.maketrans({
            "آ": "ا",  # alef madda
            "أ": "ا",  # alef hamza above
            "إ": "ا",  # alef hamza below
            "ٱ": "ا",  # alef wasla
            "ة": "ه",  # ta marbuta -> ha
            "ى": "ي",  # alef maksura -> ya
        }))

    if strip_punctuation:
        text = _PUNCT_RE.sub(" ", text)

    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Shar index discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ShardSpec:
    """One ``cuts.*.jsonl.gz`` shard belonging to a specific shar."""
    shar_dir:  str   # absolute path to the shar dir
    dataset:   str   # logical dataset name (used as composite-key prefix)
    cuts_path: str   # absolute path to the cuts.X.jsonl.gz
    recording_path: Optional[str] = None  # paired recording.X.tar (audio), if any


# Tar member suffixes that are *not* the audio payload (Lhotse pairs each cut's
# audio data member with a ``.json`` meta member; placeholders use .nodata/.nometa).
_META_EXTS = frozenset({"json", "nometa", "nodata"})


def _shard_num(path_str: str) -> Optional[str]:
    """Extract the numeric shard token from a shar filename.

    ``cuts.000007.jsonl.gz`` -> ``"000007"``; ``recording.000007.tar`` ->
    ``"000007"``.  Used to pair cuts shards with their recording tars by shard
    number rather than fragile list position.
    """
    for tok in Path(path_str).name.split("."):
        if tok.isdigit():
            return tok
    return None


def _scan_tar_offsets(tar_path: str) -> dict:
    """Header-scan a Lhotse recording tar → ``{cut_id: (offset_data, size, fmt)}``.

    Opens in seekable ``r:`` (uncompressed) mode so iterating reads only the
    512-byte member headers — no payload I/O.  ``offset_data`` + ``size`` are
    exactly the random-access pointer: ``seek(offset_data); read(size)`` returns
    the encoded audio bytes for that cut.

    A compressed tar (``.tar.gz`` etc.) cannot be randomly accessed; we log and
    return empty so the rows get null offsets (callers fall back to streaming).
    """
    import tarfile
    out: dict = {}
    try:
        tf = tarfile.open(tar_path, "r:")
    except (tarfile.ReadError, FileNotFoundError) as e:
        logger.error("Locator: cannot open %s for random access (%s) — "
                     "audio offsets will be null for this shard.", tar_path, e)
        return out
    with tf:
        for ti in tf:
            if not ti.isfile():
                continue
            name = ti.name[2:] if ti.name.startswith("./") else ti.name
            base, _, ext = name.rpartition(".")
            if not base or ext in _META_EXTS:
                continue
            out[base] = (int(ti.offset_data), int(ti.size), ext)
    return out


def _scan_tar_offsets_positional(tar_path: str) -> list:
    """Header-scan a Lhotse recording tar → ordered ``[(offset_data, size, fmt), ...]``
    for the audio members in tar order (skipping ``.json``/meta sidecars).

    Use this instead of :func:`_scan_tar_offsets` for shars whose members are
    NOT uniquely named (e.g. MLS, which names every segment by its source-file
    URL, so ~59 segments share one name). The Lhotse shar contract is
    positional — the i-th cut in ``cuts.NNNNNN.jsonl.gz`` pairs with the i-th
    audio member here — so a name-keyed dict would collapse the duplicates,
    whereas this ordered list preserves the one-cut-per-member mapping.
    """
    import tarfile
    out: list = []
    try:
        tf = tarfile.open(tar_path, "r:")
    except (tarfile.ReadError, FileNotFoundError) as e:
        logger.error("Locator(positional): cannot open %s (%s) — null offsets.", tar_path, e)
        return out
    with tf:
        for ti in tf:
            if not ti.isfile():
                continue
            name = ti.name[2:] if ti.name.startswith("./") else ti.name
            base, _, ext = name.rpartition(".")
            if not base or ext in _META_EXTS:
                continue
            out.append((int(ti.offset_data), int(ti.size), ext))
    return out


def _derive_dataset_name(shar_dir: str, dataset_root: Optional[str] = None) -> str:
    """Derive ``dataset`` tag from a shar_dir path.

    Heuristics:
      - If ``dataset_root`` is given and ``shar_dir`` is under it, use the
        relative path (e.g. shar_dir=/data/SHAR/granary_ytc/en, root=/data/SHAR
        -> "granary_ytc/en").
      - Otherwise use the last 1-2 path components (last component, or
        last2 if last is a known language code).
    """
    p = Path(shar_dir).resolve()
    if dataset_root:
        root = Path(dataset_root).resolve()
        try:
            rel = p.relative_to(root)
            return str(rel).replace(os.sep, "/")
        except ValueError:
            pass

    # Heuristic: if the parent looks like a dataset name and the leaf looks
    # like a language code, return parent/leaf; else just leaf.
    leaf = p.name
    if leaf and len(leaf) <= 4 and leaf.isalpha() and p.parent.name:
        return f"{p.parent.name}/{leaf}"
    return leaf or str(p)


def discover_shards(shar_dirs: Iterable[str], dataset_root: Optional[str] = None,
                    dataset_overrides: Optional[dict] = None,
                    audio_field: str = "recording") -> List[_ShardSpec]:
    """Walk every shar_dir, read its shar_index.json, return all cuts shards.

    Each cuts shard is paired (by shard number) with its ``{audio_field}.*.tar``
    so Stage A can header-scan the tar for per-cut random-access offsets.  A
    shar with no audio field (text-only) yields ``recording_path=None``.

    ``dataset_overrides``: optional ``{shar_dir: dataset_tag}`` mapping that
    overrides the path-based heuristic.
    """
    overrides = dataset_overrides or {}
    out: List[_ShardSpec] = []
    for sd in shar_dirs:
        sp = Path(sd).resolve()
        if not sp.is_dir():
            raise FileNotFoundError(f"Not a directory: {sd}")
        idx_path = sp / SHAR_INDEX_FILENAME
        if not idx_path.is_file():
            raise FileNotFoundError(f"Missing {SHAR_INDEX_FILENAME} in {sd}")
        with open(idx_path) as f:
            payload = json.load(f)
        fields = payload.get("fields", {})
        cuts_rel = fields.get("cuts", [])
        if not cuts_rel:
            raise ValueError(f"shar_index has no 'cuts' field: {idx_path}")

        # Pair recording tars to cuts shards by (worker-subdir, shard number).
        # Multi-worker shars repeat the same shard number across worker_XX/
        # subdirs (worker_00/recording.000000.tar, worker_01/recording.000000.tar,
        # ...), so keying on shard number alone collides and mis-pairs every cuts
        # shard with the last worker's tar -> null offsets.  Including the entry's
        # parent dir disambiguates while staying robust to list ordering.
        audio_rel = fields.get(audio_field, []) or []
        rec_by_key: dict = {}
        for ar in audio_rel:
            ap = Path(ar)
            if ap.is_absolute():
                raise ValueError(f"Absolute paths in shar_index are forbidden: {ap}")
            rec_by_key[(str(ap.parent), _shard_num(ar))] = str(sp / ap)

        ds_name = overrides.get(str(sp), _derive_dataset_name(str(sp), dataset_root))
        for r in cuts_rel:
            rp = Path(r)
            if rp.is_absolute():
                raise ValueError(f"Absolute paths in shar_index are forbidden: {rp}")
            rec_path = rec_by_key.get((str(rp.parent), _shard_num(r)))
            out.append(_ShardSpec(shar_dir=str(sp), dataset=ds_name,
                                  cuts_path=str(sp / rp),
                                  recording_path=rec_path))
    # Stable order — important for round-robin rank split determinism.
    out.sort(key=lambda s: (s.dataset, s.cuts_path))
    return out


# ---------------------------------------------------------------------------
# Per-shard worker
# ---------------------------------------------------------------------------


def _normalize_cut_id(raw: str) -> str:
    """Strip leading './' that some Shar exporters prepend."""
    if raw is None:
        return ""
    s = str(raw)
    return s[2:] if s.startswith("./") else s


def _extract_text(d: dict) -> Tuple[str, Optional[str]]:
    """Pull the supervision text + language from a Cut dict.

    Lhotse Cuts in jsonl.gz may have 'supervisions' as a list of dicts.
    Some recipes attach text directly (no supervisions); we handle both.
    """
    text = ""
    lang = None
    sups = d.get("supervisions") or []
    if sups:
        s0 = sups[0] or {}
        text = s0.get("text") or ""
        lang = s0.get("language") or None
    if not lang:
        lang = (d.get("custom") or {}).get("language") or None
    return text, lang


_ROVER_TEXT_CACHE: dict = {}


def _load_rover_text(path, field: str = "text") -> dict:
    """``{cut_id: rover[field]}`` from a rover/merged.jsonl, for
    ``manifest.text_source=rover``.

    The default field ``text`` is the reference-INDEPENDENT ASR consensus:
    identical audio yields the same key, so two cuts whose transcripts differ
    only slightly still land in one text cluster (which the original transcript
    would have split).  ``text_enhanced`` / ``text_enhanced_itn`` are also
    accepted.  Empty values are skipped.  Process-cached: each worker parses the
    (large) JSONL at most once.
    """
    key = (str(path), field)
    cached = _ROVER_TEXT_CACHE.get(key)
    if cached is not None:
        return cached
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"manifest.rover_text_path not found: {path}")
    out: dict = {}
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rv = d.get("rover") or {}
            t = rv.get(field)
            if t and str(t).strip():
                cid = d.get("cut_id") or ""
                cid = cid[2:] if cid.startswith("./") else cid
                out[cid] = t
    _ROVER_TEXT_CACHE[key] = out
    return out


_REMOVED_CACHE: dict = {}


def _get_removed_set(paths):
    """Set of already-removed ``(dataset, cut_id)`` from prior dedup runs.

    Lets an INCREMENTAL re-run (e.g. adding a new dataset to a corpus that was
    already deduped) skip cuts a previous run already dropped, instead of
    re-clustering them. Each path is a parquet with ``dataset`` + ``cut_id``
    columns (e.g. a prior retention ``assignments.parquet`` filtered to dropped
    cuts, or any removed-id list). Cached per process — the manifest Pool uses
    ``spawn``, so each worker loads once. Returns ``None`` when nothing to
    exclude."""
    if not paths:
        return None
    if isinstance(paths, str):
        paths = [paths]
    key = tuple(paths)
    cached = _REMOVED_CACHE.get(key)
    if cached is not None:
        return cached
    import pyarrow.parquet as pq
    s = set()
    for p in paths:
        try:
            t = pq.read_table(str(p), columns=["dataset", "cut_id"])
            s.update(zip(t.column("dataset").to_pylist(), t.column("cut_id").to_pylist()))
        except Exception as e:
            logger.warning("exclude_removed: could not read %s (%s)", p, e)
    _REMOVED_CACHE[key] = s
    logger.info("exclude_removed: %d already-removed (dataset,cut_id) keys loaded", len(s))
    return s


def _process_shard(args) -> dict:
    """Read one cuts.jsonl.gz, normalize, return a dict of column arrays.

    Returned dict shape (each value is a list of equal length):
        dataset, cut_id, recording_id, language, text, normalized_text,
        duration_secs, num_samples
    """
    shard, cfg = args
    cuts_path = shard.cuts_path
    dataset = shard.dataset

    strip_punct       = bool(cfg.get("strip_punctuation", False))
    min_text_chars    = int(cfg.get("min_text_chars", 4))
    drop_empty_text   = bool(cfg.get("drop_empty_text", True))
    # Incremental dedup: drop cuts already removed by a prior run (by dataset,cut_id).
    removed           = _get_removed_set(cfg.get("exclude_removed_paths"))

    # Random-access locator: header-scan the paired recording tar once, then
    # join by cut_id below.  rel_recording is stored relative to shar_dir so the
    # manifest stays portable; the Stage-D reader resolves it against shar_dir.
    # positional_audio_join: opt-in for shars with NON-UNIQUE member names (e.g.
    # MLS — every segment named by its source URL). Joins audio to cuts by tar
    # position (Lhotse's contract) instead of by name, and makes cut_ids unique
    # with a deterministic ``shard#row`` suffix. Pure manifest-side relabel — the
    # source shar is never modified.
    positional = bool(cfg.get("positional_audio_join", False)) or \
        (dataset in (cfg.get("positional_audio_join_datasets") or []))
    rel_recording = None
    if shard.recording_path:
        try:
            rel_recording = str(Path(shard.recording_path).relative_to(shard.shar_dir))
        except ValueError:
            rel_recording = Path(shard.recording_path).name
    if positional:
        offsets_list = _scan_tar_offsets_positional(shard.recording_path) if shard.recording_path else []
        offsets = {}
        shard_tag = rel_recording or Path(cuts_path).name
    else:
        offsets = _scan_tar_offsets(shard.recording_path) if shard.recording_path else {}
        offsets_list = []
        shard_tag = ""

    rows = {
        "dataset":         [],
        "cut_id":          [],
        "recording_id":    [],
        "language":        [],
        "text":            [],
        "normalized_text": [],
        "duration_secs":   [],
        "num_samples":     [],
        "tar_path":        [],
        "tar_offset":      [],
        "tar_size":        [],
        "audio_format":    [],
    }

    intra_dups: dict = {}  # cut_id -> count
    line_index = -1        # position in cuts file == position in recording tar

    with gzip.open(cuts_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_index += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Bad JSON in %s — skipping a line", cuts_path)
                continue

            cid = _normalize_cut_id(d.get("id"))
            if not cid:
                continue
            out_cid = f"{cid}::{shard_tag}#{line_index}" if positional else cid
            if removed is not None and (dataset, out_cid) in removed:
                continue   # already dropped by a prior dedup run (incremental)

            text, lang = _extract_text(d)
            # text_source=rover: cluster on the (reference-independent) ASR
            # consensus instead of the original transcript, so slight transcript
            # edits over identical audio still group together.  Falls back to the
            # supervision text when a cut has no rover entry.
            if cfg.get("text_source", "supervision") == "rover":
                rt = _load_rover_text(cfg.get("rover_text_path"),
                                      cfg.get("rover_text_field", "text")).get(cid)
                if rt:
                    text = rt
            norm = normalize_text(text, lang, strip_punctuation=strip_punct)

            if drop_empty_text and len(norm) < min_text_chars:
                continue

            # Lhotse Cuts may store recording dict, recording_id, or neither.
            rec = d.get("recording")
            if isinstance(rec, dict):
                rec_id = rec.get("id") or ""
            else:
                rec_id = d.get("recording_id") or ""

            duration = d.get("duration")
            num_samples = d.get("num_samples")
            if num_samples is None and isinstance(rec, dict):
                num_samples = rec.get("num_samples")

            rows["dataset"].append(dataset)
            rows["cut_id"].append(out_cid)
            rows["recording_id"].append(rec_id)
            rows["language"].append(lang or "")
            rows["text"].append(text)
            rows["normalized_text"].append(norm)
            rows["duration_secs"].append(float(duration) if duration is not None else None)
            rows["num_samples"].append(int(num_samples) if num_samples is not None else None)

            if positional:
                loc = offsets_list[line_index] if line_index < len(offsets_list) else None
            else:
                loc = offsets.get(cid)
            rows["tar_path"].append(rel_recording if loc is not None else None)
            rows["tar_offset"].append(loc[0] if loc is not None else None)
            rows["tar_size"].append(loc[1] if loc is not None else None)
            rows["audio_format"].append(loc[2] if loc is not None else None)

            intra_dups[out_cid] = intra_dups.get(out_cid, 0) + 1

    if positional and shard.recording_path and (line_index + 1) != len(offsets_list):
        raise ValueError(
            f"Positional join misalignment in {cuts_path}: {line_index + 1} cut "
            f"lines vs {len(offsets_list)} audio members — cannot trust audio mapping.")
    intra_dup_count = sum(1 for c in intra_dups.values() if c > 1)
    return {
        "rows": rows,
        "shard_path": cuts_path,
        "dataset": dataset,
        "intra_dup_count": intra_dup_count,
    }


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------


_SCHEMA = pa.schema([
    pa.field("dataset",         pa.string(),  nullable=False),
    pa.field("cut_id",          pa.string(),  nullable=False),
    pa.field("recording_id",    pa.string(),  nullable=True),
    pa.field("language",        pa.string(),  nullable=True),
    pa.field("text",            pa.string(),  nullable=True),
    pa.field("normalized_text", pa.string(),  nullable=False),
    pa.field("duration_secs",   pa.float64(), nullable=True),
    pa.field("num_samples",     pa.int64(),   nullable=True),
    # Random-access locator (cuDF-friendly string/int64): seek tar_offset, read
    # tar_size bytes from <shar_dir>/tar_path to fetch the encoded audio member.
    pa.field("tar_path",        pa.string(),  nullable=True),
    pa.field("tar_offset",      pa.int64(),   nullable=True),
    pa.field("tar_size",        pa.int64(),   nullable=True),
    pa.field("audio_format",    pa.string(),  nullable=True),
])


def _atomic_write_parquet(out_path: Path, table: pa.Table) -> None:
    """Write parquet via tmp + fsync + atomic rename."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    pq.write_table(
        table, str(tmp),
        compression="zstd",
        compression_level=3,
        row_group_size=100_000,
    )
    # fsync the file content + the directory entry.
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    dir_fd = os.open(str(out_path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _cleanup_tmp(dir_path: Path) -> None:
    """Delete any leftover *.parquet.tmp from a crashed previous run."""
    for p in dir_path.glob("*.parquet.tmp"):
        try:
            p.unlink()
            logger.info("Cleaned crash leftover: %s", p)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Rank-level driver
# ---------------------------------------------------------------------------


def run_rank(cfg: dict, rank: int, world_size: int) -> None:
    """Process this rank's slice of shards.  Idempotent on resume."""
    shar_dirs = cfg["shar_dirs"]
    if isinstance(shar_dirs, str):
        shar_dirs = [shar_dirs]

    output_dir = Path(cfg["output_dir"]) / "manifest"
    output_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_tmp(output_dir)

    overrides = cfg.get("dataset_overrides") or {}
    dataset_root = cfg.get("dataset_root")
    audio_field = (cfg.get("manifest") or {}).get("audio_field", "recording")

    all_shards = discover_shards(shar_dirs, dataset_root=dataset_root,
                                 dataset_overrides=overrides,
                                 audio_field=audio_field)
    my_shards = all_shards[rank::world_size]
    if not my_shards:
        # Still fall through to write an empty part — a missing part file would
        # hang the rank barrier in pipeline.py forever.
        logger.info("[rank %d/%d] no shards assigned — writing empty part",
                    rank, world_size)

    out_path = output_dir / f"part_{rank:04d}.parquet"
    if out_path.exists():
        logger.info("[rank %d/%d] %s already exists — skipping (delete to redo).",
                    rank, world_size, out_path.name)
        return

    manifest_cfg = cfg.get("manifest", {})
    if str(manifest_cfg.get("text_source", "supervision")).lower() == "rover":
        rp = manifest_cfg.get("rover_text_path")
        if not rp or not Path(rp).is_file():
            raise ValueError("manifest.text_source=rover requires an existing "
                             f"manifest.rover_text_path (got {rp!r})")
        logger.info("[rank %d/%d] text_source=rover (field=%s) from %s",
                    rank, world_size,
                    manifest_cfg.get("rover_text_field", "text"), rp)
    workers = int(manifest_cfg.get("inner_workers", min(32, len(my_shards))))
    workers = max(1, min(workers, len(my_shards)))

    logger.info("[rank %d/%d] processing %d shards with %d inner workers",
                rank, world_size, len(my_shards), workers)
    t0 = time.time()

    tasks = [(s, manifest_cfg) for s in my_shards]
    total_rows = 0
    intra_dup_total = 0

    # Convert each shard's column lists to an Arrow table as it arrives, then
    # concat_tables at the end — instead of accumulating every shard's Python
    # lists and flattening the whole rank into one giant list per column (which
    # held the rank's rows as Python objects twice).  Lower peak memory; same rows.
    tables: List[pa.Table] = []

    def _table_from_rows(rows: dict) -> pa.Table:
        return pa.Table.from_arrays(
            [pa.array(rows[name], type=_SCHEMA.field(name).type) for name in _SCHEMA.names],
            schema=_SCHEMA,
        )

    def _accumulate(res: dict) -> None:
        nonlocal total_rows, intra_dup_total
        tables.append(_table_from_rows(res["rows"]))
        total_rows += len(res["rows"]["cut_id"])
        intra_dup_total += res["intra_dup_count"]

    if workers == 1:
        for t in tasks:
            _accumulate(_process_shard(t))
    else:
        # spawn ctx so workers don't inherit any CUDA / fork-after-import state.
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            for res in pool.imap_unordered(_process_shard, tasks, chunksize=1):
                _accumulate(res)

    if intra_dup_total:
        # The plan calls for an *error* on intra-dataset cut_id duplicates;
        # here we detect them per-shard and surface them at end so a single
        # corrupt shard doesn't kill the whole job mid-flight.
        logger.error("[rank %d/%d] %d intra-shard duplicate cut_ids detected — "
                     "input shar(s) may be corrupt.", rank, world_size, intra_dup_total)

    if total_rows == 0:
        logger.warning("[rank %d/%d] no rows produced — writing empty parquet anyway",
                       rank, world_size)

    table = pa.concat_tables(tables) if tables else _SCHEMA.empty_table()
    _atomic_write_parquet(out_path, table)

    elapsed = time.time() - t0
    logger.info("[rank %d/%d] wrote %d rows to %s in %.1fs (%.0f rows/s)",
                rank, world_size, total_rows, out_path.name, elapsed,
                total_rows / max(elapsed, 1e-6))


def assert_unique_and_mark_success(cfg: dict, rank: int = 0) -> None:
    """Rank-0 only: scan every part_*.parquet, verify (dataset, cut_id) uniqueness,
    then write the _SUCCESS marker."""
    if rank != 0:
        return

    output_dir = Path(cfg["output_dir"]) / "manifest"
    parts = sorted(output_dir.glob("part_*.parquet"))
    if not parts:
        raise RuntimeError(f"No manifest parts in {output_dir}")

    seen_per_dataset: dict[str, set] = {}
    total = 0
    for p in parts:
        t = pq.read_table(p, columns=["dataset", "cut_id"])
        ds_arr  = t.column("dataset").to_pylist()
        cid_arr = t.column("cut_id").to_pylist()
        for ds, cid in zip(ds_arr, cid_arr):
            s = seen_per_dataset.setdefault(ds, set())
            if cid in s:
                raise ValueError(
                    f"Intra-dataset duplicate (dataset={ds}, cut_id={cid}) — "
                    "input shar(s) corrupt; cannot continue.")
            s.add(cid)
            total += 1

    # Cross-dataset overlap (cut_id present in ≥2 datasets): count by
    # accumulating a single occurrence map.  O(N) total.
    cid_dataset_count: dict[str, int] = {}
    for sset in seen_per_dataset.values():
        for cid in sset:
            cid_dataset_count[cid] = cid_dataset_count.get(cid, 0) + 1
    overlap = sum(1 for c in cid_dataset_count.values() if c > 1)
    logger.info("Manifest OK: %d rows across %d dataset(s); %d cut_ids appear "
                "in 2+ datasets (cross-dataset overlap, allowed).",
                total, len(seen_per_dataset), overlap)

    from . import run_layout
    run_layout.finalize_stage(output_dir.parent, "manifest", rows=total,
                              extra={"datasets": sorted(seen_per_dataset.keys())})
    logger.info("Wrote %s", output_dir / SUCCESS_MARKER)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_cfg(path: str) -> dict:
    import yaml  # PyYAML is a dependency of every other pipeline here.
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Stage A: text manifest extraction")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--rank", type=int,
                        default=int(os.environ.get("RANK",
                                    os.environ.get("SLURM_PROCID", 0))))
    parser.add_argument("--world-size", type=int,
                        default=int(os.environ.get("WORLD_SIZE",
                                    os.environ.get("SLURM_NTASKS", 1))))
    parser.add_argument("--finalize", action="store_true",
                        help="Skip processing; only verify uniqueness + write _SUCCESS.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [rank %(processName)s] %(name)s: %(message)s",
    )
    cfg = _load_cfg(args.config)
    if args.finalize:
        assert_unique_and_mark_success(cfg, rank=args.rank)
    else:
        run_rank(cfg, rank=args.rank, world_size=args.world_size)


if __name__ == "__main__":
    main()
