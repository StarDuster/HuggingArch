import json,os,subprocess,traceback,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree';CPU=W/'HuggingArch/.venv313/bin/python';GPU='/opt/sglang/bin/python'
def state(phase,**kw):
 p=R/'status.json';d=json.loads(p.read_text());d.update(state=phase,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);p.write_text(json.dumps(d,indent=2)+'\n')
def run(name,cmd,timeout=1800):
 state('running',step=name)
 with (R/(name+'.log')).open('w') as f:subprocess.run(list(map(str,cmd)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
try:
 run('source-version-tests',[GPU,'-m','pytest','-q','tests/kernel_bench/test_source_revision.py','tests/kernel_bench/test_kernel_bench.py','-k','unreleased_source_records or submodule_reports_its_packages_version or absent_module_is_still_absent'])
 run('check-source-followup',[CPU,'-m','kernel_bench','check',R/'results'])
 run('coverage',[CPU,'-m','kernel_bench','report','--jobs',R/'all-jobs.jsonl','--results',R/'results','--gpu','B300'])
 run('collect',[CPU,R/'collect.py'])
 run('estimate-impact',[CPU,R/'estimate_impact.py'],2400)
 run('save-branch',[CPU,R/'save_branch.py'])
 state('complete',step='source-followup-completed',accuracy_validated=False)
except BaseException as e:
 (R/'source-followup-error.txt').write_text(traceback.format_exc());state('failed',error=str(e));raise
