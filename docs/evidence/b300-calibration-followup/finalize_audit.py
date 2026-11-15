from pathlib import Path
import subprocess,os,json,datetime,traceback
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree';CPU=W/'HuggingArch/.venv313/bin/python'
def state(phase,**kw):
 p=R/'status.json';d=json.loads(p.read_text());d.update(state=phase,pid=os.getpid(),time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);p.write_text(json.dumps(d,indent=2)+'\n')
def run(name,args):
 state('running',step=name)
 with (R/(name+'.log')).open('w') as f:subprocess.run(list(map(str,args)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
try:
 run('collect',[CPU,R/'collect.py'])
 run('audit-overlay',[CPU,R/'audit_overlay.py'])
 # This supersedes the broad reference-overlay report as the scope-aware result.
 report=(R/'REPORT.md').read_text()
 note='## Audit follow-up\n\nSee [CALIBRATION_AUDIT.md](CALIBRATION_AUDIT.md) for the implementation audit and latest library-kernel-only partial estimate. The cross-library eager reference numbers below are diagnostic, not a production SGLang prediction. The newer source router measurements are recorded separately and included in the library-only overlay.\n\n'
 if '## Audit follow-up' not in report:
  report=report.replace('# B300 kernel calibration follow-up\n\n','# B300 kernel calibration follow-up\n\n'+note)
 (R/'REPORT.md').write_text(report)
 run('save-branch',[CPU,R/'save_branch.py'])
 state('complete',step='audited-partial-calibration-saved',accuracy_validated=False,report='CALIBRATION_AUDIT.md')
except BaseException as e:
 (R/'finalize-audit-error.txt').write_text(traceback.format_exc());state('failed',error=str(e));raise
