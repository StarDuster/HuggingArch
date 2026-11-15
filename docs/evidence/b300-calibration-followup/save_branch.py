from pathlib import Path
import subprocess,os,shutil,json,hashlib
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree';DEST=REPO/'docs/evidence/b300-calibration-followup'
files=['README.md','REPORT.md','summary.json','calibration_rows.json','new_calibration_rows.json','existing-calibration-rows.json','deduplication.json','plan-notes.json','jobs.jsonl','all-jobs.jsonl','check.log','coverage.log','estimate-impact.json','source-head.txt','source-status.txt','gpu.txt','packages.txt','branch-tests.log','source-version-diagnosis.md','measure-sglang-source-kda.log','measure-sglang-source-followup.log','source-version-tests.log','library_kernel_calibration_rows.json','overlay-audit.json','audited-comparison.json','CALIBRATION_AUDIT.md','measure-sglang-source-router.log','ENVELOPE_DIAGNOSTIC_REPORT.md','library_plus_tp8_calibration_rows.json']
files += [str(p.relative_to(R)) for p in (R/'results').glob('*.csv')]
files += [str(p.relative_to(R)) for p in (R/'tp8-communication').glob('*') if p.is_file()]
files += [str(p.relative_to(R)) for p in (R/'estimates').glob('*.summary.json')]
files += ['controlled-cold-calibration_rows.json','controlled-graph-warm-calibration_rows.json','alternative_group32_calibration_rows.json','controlled-comparison.json','attnres-trace-comparison.json','attnres-graph-result.json','weighted-attn-res-jobs.jsonl','mxfp4-grouped-jobs.jsonl','weighted-reference-tests.log','fusion-followup-tests.log','controlled-overlays.log','measure-group32-mxfp4.log','measure-weighted-attnres.log','measure-weighted-attnres-graph.log','source-followup-provenance.json','FOLLOWUP.md','status.json']
files += [str(p.relative_to(R)) for p in (R/'legacy-reference-quarantine').glob('*') if p.is_file()]
files += [str(p.relative_to(R)) for p in R.glob('*.py')]+['launch.sh']
manifest=[]
for name in files:
 src=R/name
 if not src.is_file():continue
 dst=DEST/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);manifest.append({'path':name,'bytes':src.stat().st_size,'sha256':hashlib.file_digest(src.open('rb'),'sha256').hexdigest()})
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
ENV=dict(os.environ,GH_CONFIG_DIR=str(W/'gh-config'))
def run(*args):return subprocess.run(args,cwd=REPO,env=ENV,check=True)
run('git','add','backend/inference/phase_formulas.py','backend/inference/framework_bindings/sglang.yaml','scripts/estimate_kimi_k31_sglang.py','tests/backend/inference/test_phase_formulas.py','kernel_bench/env.py','tests/kernel_bench/test_source_revision.py','tests/kernel_bench/test_source_router_scope.py','kernel_bench/providers/moe_gate.py','backend/inference/fusion_plans/kimi_attn_res_rmsnorm.yaml','backend/inference/op_signatures_catalog.py','kernel_bench/refs.py','kernel_bench/providers/gemm.py','kernel_bench/providers/norm_act.py','tests/kernel_bench/test_kimi_attn_res_reference.py','tests/backend/inference/test_fusion_registry.py','docs/evidence/b300-calibration-followup')
p=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO)
if p.returncode==1:run('git','-c','user.name=StarDuster','-c','user.email=7328584+StarDuster@users.noreply.github.com','commit','-m','Record deduplicated B300 kernel calibration and measured coverage impact')
run('git','-c','credential.helper=!'+str(W/'tools/gh')+' auth git-credential','push','https://github.com/StarDuster/HuggingArch.git','HEAD:refs/heads/codex/b300-calibration')
print('Saved branch https://github.com/StarDuster/HuggingArch/tree/codex/b300-calibration; no PR created.')
