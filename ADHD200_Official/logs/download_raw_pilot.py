import csv,json,os,time,urllib.parse,urllib.request,concurrent.futures,threading
from pathlib import Path
W=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official');P=W/'experiments/raw_pilot_18'; BASE='https://fcp-indi.s3.amazonaws.com/'
d=json.load(open(P/'qc/download_size.json'))
logpath=P/'logs/download.log';failpath=P/'logs/download_failures.csv';outpath=P/'qc/download_manifest.csv'
lock=threading.Lock(); failures=[]; results=[]
def log(s):
 with lock:
  with logpath.open('a',encoding='utf-8') as f:f.write(time.strftime('%Y-%m-%dT%H:%M:%S%z')+' '+s+'\n')

def get_file(f):
 key=f['s3_key']; dest=P/'raw'/f['local_relative_path']; dest.parent.mkdir(parents=True,exist_ok=True); part=dest.with_name(dest.name+'.part'); expected=int(f['expected_size_bytes']); url=BASE+urllib.parse.quote(key,safe='/')
 for attempt in range(1,6):
  try:
   if dest.exists():
    n=dest.stat().st_size
    if n==expected and n>0:return {**f,'local_path':str(dest),'actual_size_bytes':n,'status':'SUCCESS_ALREADY_COMPLETE','attempts':attempt-1,'failure_reason':''}
    dest.unlink()
   offset=part.stat().st_size if part.exists() else 0
   if offset>expected:part.unlink();offset=0
   headers={'Range':f'bytes={offset}-'} if offset else {}
   req=urllib.request.Request(url,headers=headers)
   with urllib.request.urlopen(req,timeout=90) as resp:
    status=resp.status
    if offset and status==206: mode='ab'
    else:
     mode='wb';offset=0
    with part.open(mode) as out:
     while True:
      block=resp.read(1024*1024)
      if not block:break
      out.write(block)
   n=part.stat().st_size if part.exists() else 0
   if n!=expected:raise IOError(f'length mismatch expected {expected} actual {n}; retained .part for resume')
   if n<=0:raise IOError('empty download')
   os.replace(part,dest)
   return {**f,'local_path':str(dest),'actual_size_bytes':n,'status':'SUCCESS','attempts':attempt,'failure_reason':''}
  except Exception as e:
   reason=f'{type(e).__name__}: {e}'
   log(f'RETRY {attempt}/5 {key} :: {reason}')
   if attempt==5:
    return {**f,'local_path':str(dest),'actual_size_bytes':part.stat().st_size if part.exists() else (dest.stat().st_size if dest.exists() else 0),'status':'FAILED','attempts':attempt,'failure_reason':reason}
   time.sleep(min(2**(attempt-1),20))
log(f'START files={d["total_files"]} bytes={d["total_bytes"]}; only selected pilot subjects and required raw BIDS objects')
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for i,res in enumerate(ex.map(get_file,d['files']),1):
  results.append(res);log(f'{res["status"]} {i}/{len(d["files"])} {res["s3_key"]} actual={res["actual_size_bytes"]}/{res["expected_size_bytes"]}')
fields=list(results[0])
with outpath.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
failed=[r for r in results if r['status']=='FAILED']
with failpath.open('w',newline='',encoding='utf-8') as f:
 cols=['s3_key','site','file_type','expected_size_bytes','actual_size_bytes','attempts','failure_reason'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows([{k:x[k] for k in cols} for x in failed])
succ=[x for x in results if x['status'].startswith('SUCCESS')]
log(f'COMPLETE successful={len(succ)}/{len(results)} failed={len(failed)} verified_bytes={sum(int(x["actual_size_bytes"]) for x in succ)}')
print(json.dumps({'files':len(results),'successful':len(succ),'failed':len(failed),'bytes_verified':sum(int(x['actual_size_bytes']) for x in succ),'failures':failed[:5]},indent=2))
