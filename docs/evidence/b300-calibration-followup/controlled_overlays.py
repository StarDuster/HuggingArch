"""Two explicitly separated AttnRes timing regimes, no latency-based filtering."""
import collections
import datetime
import json
import subprocess
import sys
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent
W = R.parents[1]
REPO = W / 'calibration-worktree'
PY = W / 'HuggingArch/.venv313/bin/python'
sys.path.insert(0, str(REPO))
from backend.inference.benchmarks import validate_text
from backend.inference.op_calibration import CalibrationRegistry, merge_validated_rows

def read_csv(name):
    _, rows = validate_text((R / name).read_text())
    assert all(r['cross_check'] == 'passed' for r in rows)
    return rows

base = json.loads((R / 'library_kernel_calibration_rows.json').read_text())
assert len(base) == 41
assert not any(r['op'] == 'attn_res_combine' for r in base)
tp8 = json.loads((R / 'tp8-communication/calibration_rows.json').read_text())
alternative = read_csv('results/flashinfer-group32-mxfp4.csv')
(R / 'alternative_group32_calibration_rows.json').write_text(json.dumps(alternative, indent=2) + '\n')
comparisons = []
historical = json.loads((W / 'benchmark/dcp-validation/results.json').read_text())['runs']
baseline = statistics.mean(r['mean_tpot_ms'] for r in historical if r['input'] == 16384)
for regime, filename in [('cold', 'results/sglang-weighted-attn-res-norm.csv'),
                         ('graph-warm', 'results/sglang-weighted-attn-res-graph-warm.csv')]:
    measured = read_csv(filename)
    assert len(measured) == 16
    rows = base + tp8 + measured
    merge_validated_rows(CalibrationRegistry(), rows, source='controlled-' + regime)
    overlay = R / ('controlled-' + regime + '-calibration_rows.json')
    overlay.write_text(json.dumps(rows, indent=2) + '\n')
    out = R / 'estimates' / ('16384-controlled-' + regime + '.json')
    status = json.loads((R / 'status.json').read_text())
    status.update(step='estimate-controlled-' + regime, active_work=True,
                  updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  remaining_gaps=['selected TRTLLM full MoE dispatch', 'remaining unfused AttnRes boundaries',
                                  'serving cache/PDL overlap versus isolated measurements'])
    (R / 'status.json').write_text(json.dumps(status, indent=2) + '\n')
    print('Estimating', regime, len(rows), 'rows', flush=True)
    with (R / ('estimate-controlled-' + regime + '.log')).open('w') as log:
        subprocess.run([str(PY), str(REPO / 'scripts/estimate_kimi_k31_sglang.py'),
                        '--dcp', '8', '--input-tokens', '16384', '--kv-block-size', '64',
                        '--calibration-rows', str(overlay), '--output', str(out)],
                       cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
    summary = json.loads(out.with_suffix('.summary.json').read_text())['phases']['decode']
    detail = json.loads(out.read_text())
    by_op = collections.defaultdict(lambda: dict(aggregated_rows=0, calls_per_step=0, measured_calls_per_step=0, partial_ms=0.0))
    for row in detail['per_op_timing_decode']:
        entry = by_op[row['op']]
        entry['aggregated_rows'] += 1
        entry['calls_per_step'] += row['count_per_step']
        if row.get('t_measured_s') is not None:
            entry['measured_calls_per_step'] += row['count_per_step']
        entry['partial_ms'] += 1000 * (row.get('t_measured_s') if row.get('t_measured_s') is not None else row.get('t_sol_s', 0))
    item = dict(regime=regime, overlay_rows=len(rows), input_tokens=16384,
                partial_ms=summary['t_partial_s'] * 1000,
                historical_tpot_ms=baseline,
                error_pct=100 * (summary['t_partial_s'] * 1000 / baseline - 1),
                partial_measured_share=summary['partial_measured_share'],
                fully_measured_s=summary['t_measured_s'], by_op=dict(by_op))
    comparisons.append(item)
    (R / 'controlled-comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
    print(json.dumps({k: v for k, v in item.items() if k != 'by_op'}), flush=True)
print('Both regimes are partial mixed-clock diagnostics, not a full serving dispatch reconstruction.', flush=True)
