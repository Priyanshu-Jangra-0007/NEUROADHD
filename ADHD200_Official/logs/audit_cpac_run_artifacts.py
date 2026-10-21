import urllib.parse,urllib.request,xml.etree.ElementTree as ET,json,os,concurrent.futures,collections
from pathlib import Path
W=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official');B='https://fcp-indi.s3.amazonaws.com/';ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
PREFIX='data/Projects/ADHD200/Outputs/cpac/'
def listall(prefix,delimiter=None):
 out=[];token=None;common=[]
 while True:
  q={'list-type':'2','prefix':prefix,'max-keys':'1000'}
  if delimiter:q['delimiter']=delimiter
  if token:q['continuation-token']=token
  with urllib.request.urlopen(B+'?'+urllib.parse.urlencode(q),timeout=60) as r:x=ET.fromstring(r.read())
  out.extend([{'key':e.findtext('s:Key',namespaces=ns),'size_bytes':int(e.findtext('s:Size',default='0',namespaces=ns) or 0),'etag':(e.findtext('s:ETag',namespaces=ns) or '').strip('"'),'last_modified':e.findtext('s:LastModified',namespaces=ns)} for e in x.findall('s:Contents',ns)])
  common.extend([e.findtext('s:Prefix',namespaces=ns) for e in x.findall('s:CommonPrefixes',ns)])
  if x.findtext('s:IsTruncated',namespaces=ns)!='true':break
  token=x.findtext('s:NextContinuationToken',namespaces=ns)
 return out,common
root,dirs=listall(PREFIX,delimiter='/')
out={'root_dirs':dirs,'roots':{},'logs':[],'crashes':[],'working_metadata_candidates':[]}
for d in dirs:
 name=d[len(PREFIX):].strip('/')
 if name in ('logs','crashes','working'):
  objs,subs=listall(d)
  out['roots'][name]={'object_count':len(objs),'total_bytes':sum(x['size_bytes'] for x in objs),'subprefix_count':len(subs),'subprefix_examples':subs[:20]}
  if name in ('logs','crashes'):
   # all objects under these roots are execution diagnostics; GET only bounded small text-like logs/crash records <=100KB.
   out[name]=objs
  else:
   out['working_metadata_candidates']=[x for x in objs if any(t in x['key'].lower() for t in ('config','pipeline','workflow','provenance','version','log','crash','nodeblock')) or os.path.splitext(x['key'].lower())[1] in ('.yml','.yaml','.json','.log','.txt','.pklz')][:10000]
# Identify job logs and crash files and fetch only text-like artifacts <=20 KB; never fetch derivative NIfTI/ROI timeseries.
get_targets=[]
for rootname in ('logs','crashes'):
 for o in out[rootname]:
  k=o['key'].lower()
  if o['size_bytes']<=20000 and (rootname=='crashes' or k.endswith(('.log','.txt','.e1','.o1')) or '/run_cpac.' in k):get_targets.append(o)
def get(o):
 with urllib.request.urlopen(B+urllib.parse.quote(o['key'],safe='/'),timeout=40) as r:return o['key'],r.read().decode('utf-8','replace')
texts={};errors={}
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:
 fs={ex.submit(get,o):o for o in get_targets}
 for f in concurrent.futures.as_completed(fs):
  try:k,v=f.result();texts[k]=v
  except Exception as e:errors[fs[f]['key']]=str(e)
# Save fetched small logs as evidence corpus in the dedicated audit folder, preserving key path only as metadata text.
(W/'metadata/cpac_s3_execution_artifacts.json').write_text(json.dumps({'source':'Anonymous official S3 list plus GET of small text logs/crash artifacts only; no imaging/derivative files were downloaded.','root_dirs':dirs,'roots':out['roots'],'logs':out['logs'],'crashes':out['crashes'],'working_metadata_candidates':out['working_metadata_candidates'],'fetched_text_artifacts':texts,'fetch_errors':errors},indent=2)+'\n')
# Print useful, non-huge summary and all matched evidence excerpts/keys.
terms=('cpac','python','version','docker','singularity','apptainer','pipeline','command','config','workflow','nipype','afni','fsl','ants','error','failed','crash','module','argv','usage')
print(json.dumps({'root_dirs':dirs,'roots':out['roots'],'fetch_targets':len(get_targets),'fetched':len(texts),'fetch_errors':len(errors),'working_metadata_candidates':len(out['working_metadata_candidates'])},indent=2))
for k,v in texts.items():
 if any(t in v.lower() for t in terms) or '/run_cpac.' in k:
  print('\n---',k,'---\n',v[:7000])
