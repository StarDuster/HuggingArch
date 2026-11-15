import json,subprocess,time,os,datetime,traceback
from pathlib import Path
R=Path(__file__).resolve().parent;W=R.parents[1];REPO=W/'calibration-worktree'
CPU=W/'HuggingArch/.venv313/bin/python';GPU=Path('/opt/sglang/bin/python')
steps=[]
def state(phase,**kw):
 d=dict(state=phase,time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),steps=steps,**kw);p=R/'status.tmp';p.write_text(json.dumps(d,indent=2)+'\n');p.replace(R/'status.json');print(json.dumps(d),flush=True)
def run(name,args,timeout=1800):
 state('running',step=name)
 with (R/(name+'.log')).open('w') as f:
  try:p=subprocess.run(list(map(str,args)),cwd=REPO,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);rc=p.returncode
  except subprocess.TimeoutExpired:rc=124
 steps.append(dict(name=name,returncode=rc));return rc
try:
 state('preparing')
 # Re-read omp session at execution time; preserve only user/assistant prose, not hidden reasoning.
 lines=[]
 sessions=Path('/root/.omp/agent/sessions/-tmp-scratch-space-k3-performance')
 for path in sorted(sessions.glob('*.jsonl')):
  if path.name.startswith('2026-11-15T00-43-15'):
   for raw in path.open():
    try:d=json.loads(raw)
    except json.JSONDecodeError:continue
    m=d.get('message',{})
    if m.get('role') in ['assistant','user']:
     for c in m.get('content',[]):
      if isinstance(c,dict) and c.get('type')=='text':lines.append(m['role']+': '+c['text'])
 (R/'omp-session-context.txt').write_text('\n\n'.join(lines))
 for key,args in {'source-head':['git','rev-parse','HEAD'],'source-status':['git','status','--short'],'gpu':['nvidia-smi','--query-gpu=index,name,uuid,memory.used,utilization.gpu,driver_version','--format=csv'],'packages':[str(GPU),'-m','pip','freeze']}.items():
  p=subprocess.run(args,cwd=REPO,capture_output=True,text=True);(R/(key+'.txt')).write_text(p.stdout+p.stderr)
 assert run('refresh-existing',['git','-C',W/'kimi-k31-reports','-c','credential.helper=!'+str(W/'tools/gh')+' auth git-credential','fetch','origin','kimi-k31-reports'])==0,'Cannot refresh calibration branch'
 assert run('prepare',[CPU,R/'prepare.py'],1800)==0,'Plan failed'
 (R/'results').mkdir(exist_ok=True)
 for library in ['torch','deepgemm','flashinfer','sglang']:
  run('measure-'+library,[GPU,'-m','kernel_bench','run','--jobs',R/'jobs.jsonl','-o',R/f'results/{library}.csv','--library',library,'--contributor','StarDuster B300 calibration','--box','b300-followup','--run-id','b300-followup-'+library],3600)
 run('check',[CPU,'-m','kernel_bench','check',R/'results'])
 run('coverage',[CPU,'-m','kernel_bench','report','--jobs',R/'all-jobs.jsonl','--results',R/'results','--gpu','B300'])
 assert run('collect',[CPU,R/'collect.py'])==0,'No valid calibration rows'
 assert run('estimate-impact',[CPU,R/'estimate_impact.py'],3600)==0,'Estimate impact analysis failed'
 assert run('save-branch',[CPU,R/'save_branch.py'],1800)==0,'Saving calibration branch failed'
 state('complete',summary='summary.json',calibration_rows='calibration_rows.json')
except BaseException as e:
 (R/'controller-error.txt').write_text(traceback.format_exc());state('failed',error=str(e));raise
