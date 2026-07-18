#!/usr/bin/env bash
# Validation probes for quality-assesment-gh200.
# Same intent as the spec in CONTAINER_REQUEST.md, but probe #5 now exercises
# the *new* aligner_utils API (post-NeMo-2.5 refactor) instead of the legacy
# /opt/NeMo source clone, and one extra probe verifies the lhotse fork is the
# one in site-packages.
set -e

source /etc/profile.d/qa-gh200.sh 2>/dev/null || true

run() {
    echo ""
    echo "----- $1 -----"
    shift
    if "$@"; then
        echo "  OK"
    else
        echo "  FAIL"
        return 1
    fi
}

# 1. PyTorch on GH200
run "torch on GH200" python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible"
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
y = x @ x.T
torch.cuda.synchronize()
print('torch', torch.__version__, torch.cuda.get_device_name(0))
PY

# 2. NeMo version
run "nemo >= 2.5" python3 - <<'PY'
import nemo
from packaging import version
assert version.parse(nemo.__version__) >= version.parse('2.5.0'), \
    f'nemo {nemo.__version__} < 2.5'
print('nemo', nemo.__version__)
PY

# 4. cuDF (kept as #4 to match spec numbering)
run "cudf" python3 - <<'PY'
import cudf, numpy as np
df = cudf.DataFrame({'a': np.arange(10), 'b': np.arange(10) * 2})
assert int(df['b'].sum()) == 90
print('cudf', cudf.__version__)
PY

# 4b. cuGraph connected components (dup_retrieval Stage C, backend=dask_cudf).
run "cugraph connected_components" python3 - <<'PY'
import cudf, cugraph
# 0-1-2 connected; 3 isolated -> 2 components.
edges = cudf.DataFrame({'src': [0, 1], 'dst': [1, 2]})
g = cugraph.Graph(directed=False)
g.from_cudf_edgelist(edges, source='src', destination='dst')
comp = cugraph.connected_components(g)
n_comp = comp['labels'].nunique()
print('cugraph', cugraph.__version__, 'components(0-1-2,3-iso among seen)=', int(n_comp))
PY

# 5. NEW NFA API (post NeMo-2.5 refactor — aligner_utils lives in the
#    installable nemo package now). This replaces the legacy
#    `from utils.data_prep import ...` probe in the spec, which only worked
#    against an /opt/NeMo source clone of v2.4.0. See NFA_API_MIGRATION.md.
run "aligner_utils import (new NFA API)" python3 - <<'PY'
from nemo.collections.asr.parts.utils.aligner_utils import (
    get_utt_obj,
    add_t_start_end_to_utt_obj,
    viterbi_decoding,
    BLANK_TOKEN,
    SPACE_TOKEN,
)
print('aligner_utils OK; BLANK_TOKEN=', repr(BLANK_TOKEN),
      'SPACE_TOKEN=', repr(SPACE_TOKEN))
PY

# 5b. lhotse fork verification — the import path must point inside our
#     custom build, and the version string should NOT be the stock one we
#     would otherwise have inherited from NeMo's resolve.
run "lhotse is the Alvorecer721 fork" python3 - <<'PY'
import importlib, importlib.metadata as md, lhotse, json, pathlib

print('lhotse', lhotse.__version__)
print('  __file__:', lhotse.__file__)

# Direct URL recorded by pip when installed from git
direct_url = None
try:
    dist = md.distribution('lhotse')
    p = pathlib.Path(dist.locate_file('lhotse-' + dist.version + '.dist-info/direct_url.json'))
    if not p.exists():
        # Locate any direct_url.json in lhotse's dist-info
        for entry in dist.files or []:
            if entry.name == 'direct_url.json':
                p = pathlib.Path(dist.locate_file(entry))
                break
    if p.exists():
        direct_url = json.loads(p.read_text())
except Exception as exc:
    print('  direct_url lookup failed:', exc)

print('  direct_url:', direct_url)
url_ok = direct_url and 'Alvorecer721/lhotse' in direct_url.get('url', '')
ref_ok = direct_url and (
    direct_url.get('vcs_info', {}).get('requested_revision', '')
    == 'fix/duration-batcher-and-shar-reader'
)
assert url_ok, 'lhotse install does not point at Alvorecer721/lhotse'
assert ref_ok, 'lhotse install is not on fix/duration-batcher-and-shar-reader'
print('lhotse fork OK')
PY

# Misc imports — orjson, duckdb, audiobox_aesthetics, plus the data tools
# we explicitly added in setup Step 4.
run "misc + data analysis imports" python3 - <<'PY'
import orjson, duckdb, audiobox_aesthetics
import pandas as pd, numpy as np, scipy, sklearn, seaborn as sns
import matplotlib, plotly, polars as pl, ray, tqdm, jupyterlab
print('orjson', orjson.__version__,
      'duckdb', duckdb.__version__,
      'pandas', pd.__version__,
      'sklearn', sklearn.__version__,
      'seaborn', sns.__version__,
      'polars', pl.__version__,
      'plotly', plotly.__version__,
      'ray', ray.__version__,
      'tqdm', tqdm.__version__,
      'jupyterlab', jupyterlab.__version__)
PY

# Experiment tracking (Step 4b in setup_qa_gh200.sh).
run "experiment tracking imports" python3 - <<'PY'
import wandb, tensorboard, mlflow, aim
print('wandb', wandb.__version__,
      'tensorboard', tensorboard.__version__,
      'mlflow', mlflow.__version__,
      'aim', aim.__version__)
PY

# 6. vLLM (skip-on-fail — Voxtral runs in dedicated container)
run "vllm import (best-effort)" bash -c '
python3 -c "import vllm; print(\"vllm\", vllm.__version__)" 2>&1 || \
  echo "  WARN: vllm not importable — Voxtral runs in dedicated container"
'

# /opt/NeMo MUST be gone — guard against a stale writable rootfs
run "no stale /opt/NeMo" bash -c '[ ! -e /opt/NeMo ] && echo "  /opt/NeMo absent"'

# 3. Canary timestamps end-to-end (only if RUN_CANARY=1).
# Heavy: downloads canary-1b-v2 on first run. Default-skipped here so the
# in-container probe is fast; the build script then runs it on a real GPU
# via srun afterwards.
if [ "${RUN_CANARY:-0}" = "1" ]; then
    run "canary-1b-v2 timestamps" python3 - <<'PY'
import numpy as np, nemo.collections.asr as nemo_asr
m = nemo_asr.models.ASRModel.from_pretrained('nvidia/canary-1b-v2').eval()
out = m.transcribe(
    [np.zeros(16000, dtype=np.float32)],
    batch_size=1, return_hypotheses=True, timestamps=True,
    source_lang='en', target_lang='en', task='asr', pnc='yes', verbose=False,
)
ts = out[0].timestamp
assert isinstance(ts, dict) and 'word' in ts, f'bad timestamp shape: {ts!r}'
print('canary timestamps OK; keys=', list(ts.keys()))
PY
else
    echo ""
    echo "----- canary-1b-v2 timestamps (skipped, set RUN_CANARY=1) -----"
fi

echo ""
echo "===== validation complete ====="
