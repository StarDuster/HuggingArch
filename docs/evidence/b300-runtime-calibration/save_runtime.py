"""Save reviewable source/evidence to the existing fork branch, never a PR."""
import datetime,hashlib,json,os,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree'
DEST=REPO/'docs/evidence/b300-runtime-calibration';DEST.mkdir(parents=True,exist_ok=True)
archive=R.parent/'b300-runtime-calibration-20261115.tar.gz'
(R/'private-release.json').write_text(json.dumps(dict(
    url='https://github.com/StarDuster/k3-performance-results/releases/tag/b300-runtime-calibration-20261115',
    archive=archive.name,bytes=archive.stat().st_size,
    sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()),indent=2)+'\n')
expanded={'prediction-16384.json','prediction-2048.json','whole-prediction-16384.json','whole-prediction-2048.json'}
manifest=[]
for src in sorted(R.rglob('*')):
    if not src.is_file() or '__pycache__' in src.parts or src.name in expanded:continue
    relative=src.relative_to(R);dst=DEST/relative;dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)
    manifest.append(dict(path=str(relative),bytes=src.stat().st_size,
        sha256=hashlib.file_digest(src.open('rb'),'sha256').hexdigest()))
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
prior=W/'benchmark/b300-calibration-followup'
shutil.copyfile(prior/'REPORT.md',REPO/'docs/evidence/b300-calibration-followup/REPORT.md')
status=json.loads((prior/'status.json').read_text())
status.update(state='runtime-calibration-validated',active_work=False,background_jobs=[],
    step='saved-runtime-repair',updated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    accuracy_validated=True,accuracy_scope='Historical target-only B1, 2K/16K, absolute error below8%; partial model, not AgentX',
    fully_measured=False,final_results=json.loads((R/'whole-comparison.json').read_text()),
    report='../b300-runtime-calibration/REPORT.md')
(prior/'status.json').write_text(json.dumps(status,indent=2)+'\n')
shutil.copyfile(prior/'status.json',REPO/'docs/evidence/b300-calibration-followup/status.json')
source_files=list(json.loads((R/'SUMMARY.json').read_text())['huggingarch_source_sha256'])
env=dict(os.environ,GH_CONFIG_DIR=str(W/'gh-config'))
def run(*args):subprocess.run(args,cwd=REPO,env=env,check=True)
run('git','diff','--check','--',*source_files)
run('git','add',*source_files,'docs/evidence/b300-runtime-calibration',
    'docs/evidence/b300-calibration-followup/REPORT.md','docs/evidence/b300-calibration-followup/status.json')
run('git','-c','user.name=StarDuster','-c','user.email=7328584+StarDuster@users.noreply.github.com',
    'commit','-m','Calibrate K3.1 TP8 execution spans with distributed MoE timing')
run('git','-c','credential.helper=!'+str(W/'tools/gh')+' auth git-credential',
    'push','https://github.com/StarDuster/HuggingArch.git','HEAD:refs/heads/codex/b300-calibration')
print('Saved source and evidence; no PR created.',flush=True)
