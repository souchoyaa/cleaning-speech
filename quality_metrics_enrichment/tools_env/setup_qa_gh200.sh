#!/usr/bin/env bash
# =============================================================================
# setup_qa_gh200.sh — install pipeline deps on top of nemo-curator+cuda-fix base
#
# Adds on top of the base image (which already has PyTorch 2.9 cu128, NeMo 2.4,
# cuDF/cuML 25.10, lhotse, librosa, omegaconf, pyarrow, etc.):
#
#   - NeMo >= 2.5  (canary-1b-v2 timestamps land in 2.5; NFA helpers move to
#                   nemo.collections.asr.parts.utils.aligner_utils)
#   - lhotse   from Alvorecer721 fork, branch fix/duration-batcher-and-shar-reader
#              (force-replaces the lhotse pulled in by NeMo)
#   - duckdb, audiobox_aesthetics, orjson
#   - Data analysis tools: tqdm, pandas, ray, scikit-learn, seaborn, polars,
#     plotly, jupyterlab, ipykernel  (most are already in the base; this only
#     installs the missing ones)
#   - Experiment tracking / observability: wandb, tensorboard, mlflow, aim
#     (wandb + tensorboard are already in the base, pinned here so a future
#     base change can't silently drop them)
#   - vLLM aarch64 (best-effort)
#
# /opt/NeMo is intentionally NOT cloned anymore. The legacy NFA helpers
# (get_utt_obj / add_t_start_end_to_utt_obj / viterbi_decoding) live in the
# installed nemo package now (see tools/NFA_API_MIGRATION.md).
#
# ENV vars:
#   QA_NEMO_VERSION    pip spec, default ">=2.5,<2.8"
#   QA_VLLM_VERSION    pip spec for vllm, default "0.13.0". Empty to skip.
#   QA_LHOTSE_GIT      override lhotse git URL/ref to install
# =============================================================================

set -e
set -o pipefail

QA_NEMO_VERSION="${QA_NEMO_VERSION:-nemo-toolkit[asr]>=2.5,<2.8}"
QA_VLLM_VERSION="${QA_VLLM_VERSION:-0.13.0}"
QA_LHOTSE_GIT="${QA_LHOTSE_GIT:-git+https://github.com/Alvorecer721/lhotse.git@fix/duration-batcher-and-shar-reader}"

echo "=== qa setup: python / pip ==="
python3 --version
pip3 --version

echo ""
echo "=== Step 1: misc python deps ==="
pip3 install --no-cache-dir \
    "duckdb>=1.0,<2.0" \
    "orjson>=3.10,<4.0" \
    "audiobox_aesthetics==0.0.4" \
    "num2words" \
    "rapidfuzz"

echo ""
echo "=== Step 2: upgrade nemo-toolkit ==="
# Pin upper bound to <2.8 so we don't accidentally jump to a release that
# hasn't been validated against the rest of the stack. NeMo will pull lhotse
# 1.33.0 — we override that with the project-specific fork in Step 3.
pip3 install --no-cache-dir --upgrade "${QA_NEMO_VERSION}"

# Sanity print
python3 -c "import nemo, packaging.version as v; \
  assert v.parse(nemo.__version__) >= v.parse('2.5.0'), nemo.__version__; \
  print('nemo', nemo.__version__)"

echo ""
echo "=== Step 3: lhotse fork (Alvorecer721/fix/duration-batcher-and-shar-reader) ==="
# --force-reinstall + --no-deps: keep lhotse's transitive deps from NeMo's resolve,
# but make sure the lhotse code itself is the fork, not whatever NeMo pulled.
pip3 install --no-cache-dir --force-reinstall --no-deps "${QA_LHOTSE_GIT}"
python3 -c "import lhotse; print('lhotse', lhotse.__version__, 'from', lhotse.__file__)"

echo ""
echo "=== Step 3b: cuGraph (multi-GPU connected components for dup_retrieval Stage C) ==="
# RAPIDS cuGraph, pinned to the EXACT cudf version already in the base so the
# RAPIDS stack stays consistent (cudf/cuml/raft are 25.10; cuGraph must match).
# Used by dup_retrieval text_dedup backend=dask_cudf for GPU/multi-GPU
# connected components.  Verified to resolve cleanly on aarch64 (deps already
# satisfied by the installed cudf 25.10).
QA_CUGRAPH_VERSION="${QA_CUGRAPH_VERSION:-$(python3 -c 'import cudf,re;print(re.match(r"\d+\.\d+",cudf.__version__).group(0))')}"
pip3 install --no-cache-dir "cugraph-cu12==${QA_CUGRAPH_VERSION}.*"
python3 -c "import cugraph; print('cugraph', cugraph.__version__)"

echo ""
echo "=== Step 4: data analysis tools (only what's missing in the base) ==="
# scikit-learn, seaborn, polars, plotly, jupyterlab, ipykernel are not in the
# base; pandas/ray/tqdm/matplotlib/numpy/scipy ARE — pip will skip them.
pip3 install --no-cache-dir \
    "scikit-learn" \
    "seaborn" \
    "polars" \
    "plotly" \
    "jupyterlab" \
    "ipykernel" \
    "pyarrow"   # ensure pinned; base already has it but harmless

echo ""
echo "=== Step 4b: experiment tracking / observability ==="
# wandb (0.23.x) and tensorboard (2.20.x) are already pulled in by the base
# (NeMo / jupyterlab transitive deps); pin them explicitly so a future base
# image change can't silently drop them. mlflow + aim are not in the base.
pip3 install --no-cache-dir \
    "wandb>=0.18,<1.0" \
    "tensorboard>=2.18,<3.0" \
    "mlflow>=2.16,<4.0" \
    "aim>=3.25,<4.0"

echo ""
echo "=== Step 5: vLLM (best-effort) ==="
if [ -n "${QA_VLLM_VERSION}" ]; then
    if pip3 install --no-cache-dir "vllm==${QA_VLLM_VERSION}"; then
        echo "vllm ${QA_VLLM_VERSION} install OK"
    else
        echo "WARNING: vllm ${QA_VLLM_VERSION} pip install failed on aarch64."
        echo "Voxtral serving uses the dedicated nemo+vllm container on a separate"
        echo "node, so the rest of the pipeline still works without vllm in this image."
    fi
else
    echo "QA_VLLM_VERSION empty — skipping vllm install"
fi

echo ""
echo "=== Step 6: cleanup any /opt/NeMo from a prior build ==="
# This was used in the very first build of this image to expose the legacy
# tools/nemo_forced_aligner package. The new aligner_utils module makes it
# obsolete (see tools/NFA_API_MIGRATION.md).
if [ -e /opt/NeMo ]; then
    rm -rf /opt/NeMo
    echo "  removed /opt/NeMo"
else
    echo "  /opt/NeMo not present"
fi

echo ""
echo "=== Step 7: bake env defaults ==="
cat > /etc/profile.d/qa-gh200.sh <<'PROFILE'
# GH200 PyTorch allocator hint (torch 2.9 renamed PYTORCH_CUDA_ALLOC_CONF
# → PYTORCH_ALLOC_CONF; the old name still works but emits a deprecation
# warning at every CUDA init).
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# Avoid threading-on-import storms
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
PROFILE
chmod 0644 /etc/profile.d/qa-gh200.sh

echo ""
echo "=== qa setup: done ==="
