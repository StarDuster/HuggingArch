import json,subprocess,time,os
from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'agentx-yyh/version-1.0.14-seed42/status.json'
def state(phase,**kw):
 d=dict(state=phase,time=time.time(),pid=os.getpid(),depends_on=str(P),**kw);t=R/'queue-status.tmp';t.write_text(json.dumps(d,indent=2)+'\n');t.replace(R/'queue-status.json')
idle=None
while True:
 try:
  s=json.loads(P.read_text()) if P.exists() else {}
  cancelled=(P.parent/'cancelled-by-user.json').exists()
  if s.get('state')=='failed' and not cancelled:state('blocked_agentx_failed',error=s.get('error'));break
  if not cancelled and (s.get('state')!='complete' or sorted(s.get('completed',[]))!=[4,8,16,32,64] or not s.get('resources_released')):
   state('waiting_for_agentx_completion',completed=s.get('completed',[]));idle=None;time.sleep(30);continue
  p=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
  rows=[list(map(int,line.split(','))) for line in p.stdout.splitlines()] if p.returncode==0 else []
  free=len(rows)==8 and all(mem==0 and util==0 for mem,util in rows);idle=(idle or time.time()) if free else None
  state('waiting_for_gpu_release',gpu=rows,idle_since=idle)
  if idle and time.time()-idle>=120:
   with (R/'controller.log').open('w') as f:p=subprocess.Popen(['bash',str(R/'launch.sh')],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   state('launched',controller_pid=p.pid);break
 except Exception as e:state('waiting_check_error',error=str(e));idle=None
 time.sleep(30)
