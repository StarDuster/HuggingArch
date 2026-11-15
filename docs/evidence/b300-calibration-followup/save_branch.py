from pathlib import Path
import subprocess,os,shutil,json,hashlib
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree';DEST=REPO/'docs/evidence/b300-calibration-followup'
files=['README.md','REPORT.md','summary.json','calibration_rows.json','new_calibration_rows.json','existing-calibration-rows.json','deduplication.json','plan-notes.json','jobs.jsonl','all-jobs.jsonl','check.log','coverage.log','estimate-impact.json','source-head.txt','source-status.txt','gpu.txt','packages.txt','branch-tests.log']
files += [str(p.relative_to(R)) for p in (R/'results').glob('*.csv')]
files += [str(p.relative_to(R)) for p in (R/'estimates').glob('*.summary.json')]
files += [str(p.relative_to(R)) for p in R.glob('*.py')]+['launch.sh']
manifest=[]
for name in files:
 src=R/name
 if not src.is_file():continue
 dst=DEST/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);manifest.append({'path':name,'bytes':src.stat().st_size,'sha256':hashlib.file_digest(src.open('rb'),'sha256').hexdigest()})
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
ENV=dict(os.environ,GH_CONFIG_DIR=str(W/'gh-config'))
def run(*args):return subprocess.run(args,cwd=REPO,env=ENV,check=True)
run('git','add','backend/inference/phase_formulas.py','backend/inference/framework_bindings/sglang.yaml','scripts/estimate_kimi_k31_sglang.py','tests/backend/inference/test_phase_formulas.py','docs/evidence/b300-calibration-followup')
p=subprocess.run(['git','diff','--cached','--quiet'],cwd=REPO)
if p.returncode==1:run('git','-c','user.name=StarDuster','-c','user.email=7328584+StarDuster@users.noreply.github.com','commit','-m','Record deduplicated B300 kernel calibration and measured coverage impact')
run('git','-c','credential.helper=!'+str(W/'tools/gh')+' auth git-credential','push','https://github.com/StarDuster/HuggingArch.git','HEAD:refs/heads/codex/b300-calibration')
print('Saved branch https://github.com/StarDuster/HuggingArch/tree/codex/b300-calibration; no PR created.')
