import csv,json,urllib.parse,urllib.request,xml.etree.ElementTree as ET,concurrent.futures,collections,os,time
from pathlib import Path
W=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official'); NP=Path('/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD')
BUCKET='https://fcp-indi.s3.amazonaws.com/'; ROOT='data/Projects/ADHD200/Outputs/cpac/raw_outputs/pipeline_adhd200-benchmark__freq-filter/'; ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
with (NP/'data/adhd200/features/cc200/cc200_connectivity_features.csv').open(newline='',encoding='utf-8') as f: ids=sorted({r['subject_id'].zfill(7) for r in csv.DictReader(f)})
# Include the one labeled input excluded on constant-signal QC to retain derivative provenance context.
ids.append('0015038'); ids=sorted(set(ids))
def get_list(prefix):
 out=[];token=None
 while True:
  q={'list-type':'2','prefix':prefix,'max-keys':'1000'}
  if token:q['continuation-token']=token
  with urllib.request.urlopen(BUCKET+'?'+urllib.parse.urlencode(q),timeout=60) as r:xml=ET.fromstring(r.read())
  for e in xml.findall('s:Contents',ns):
   out.append({'key':e.findtext('s:Key',namespaces=ns),'size_bytes':int(e.findtext('s:Size',default='0',namespaces=ns) or 0),'etag':(e.findtext('s:ETag',namespaces=ns) or '').strip('"'),'last_modified':e.findtext('s:LastModified',namespaces=ns)})
  if xml.findtext('s:IsTruncated',namespaces=ns)!='true':break
  token=xml.findtext('s:NextContinuationToken',namespaces=ns)
 return out
# Root delimiter listing shows top-level artifacts without recursing over all pipeline objects.
def list_root():
 q={'list-type':'2','prefix':ROOT,'delimiter':'/','max-keys':'1000'}
 with urllib.request.urlopen(BUCKET+'?'+urllib.parse.urlencode(q),timeout=60) as r:xml=ET.fromstring(r.read())
 return {'contents':[e.findtext('s:Key',namespaces=ns) for e in xml.findall('s:Contents',ns)],'prefixes':[e.findtext('s:Prefix',namespaces=ns) for e in xml.findall('s:CommonPrefixes',ns)]}
root=list_root();out=[];errs={}
def work(sid):return sid,get_list(ROOT+sid+'_session_1/')
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
 fs={ex.submit(work,sid):sid for sid in ids}
 for f in concurrent.futures.as_completed(fs):
  sid=fs[f]
  try:_,x=f.result();out.extend([{'subject_id':sid,**o} for o in x])
  except Exception as e:errs[sid]=f'{type(e).__name__}: {e}'
 out.sort(key=lambda x:x['key'])
with (W/'metadata/cpac_existing153_s3_object_inventory.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=['subject_id','key','size_bytes','etag','last_modified']);w.writeheader();w.writerows(out)
# Metadata-like artifact names only; list object bodies not retrieved.
keywords=('pipeline','config','workflow','provenance','version','environment','crash','log','command','cmdline','nipype','cpac','docker','container','singularity','apptainer')
metadata=[]
for o in out:
 k=o['key'];base=os.path.basename(k).lower();ext=os.path.splitext(base)[1]
 if ext in ('.yml','.yaml','.json','.log','.txt','.pklz','.pickle','.html','.xml','.jsonl','.tsv') or any(x in base for x in keywords):metadata.append(o)
summary={'source':'Anonymous S3 ListObjectsV2 metadata only; no object GET or derivative payload downloaded by this audit.','pipeline_prefix':'s3://fcp-indi/'+ROOT,'subjects_queried':len(ids),'subjects_current_feature_cohort':len(ids)-1,'additional_known_excluded_subjects':['0015038'],'subject_prefix_listing_errors':errs,'objects_listed':len(out),'metadata_like_objects':len(metadata),'root_listing':root,'metadata_like':metadata,'all_objects_csv':'metadata/cpac_existing153_s3_object_inventory.csv','metadata_like_classifications':dict(collections.Counter(os.path.splitext(os.path.basename(x['key']).lower())[1] for x in metadata)),'note':'Metadata-like selection is filename/extension based; it does not prove these files are C-PAC workflow provenance. Object contents were not retrieved.'}
(W/'metadata/cpac_existing153_s3_listing_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ('pipeline_prefix','subjects_queried','subject_prefix_listing_errors','objects_listed','metadata_like_objects','metadata_like_classifications','root_listing')},indent=2))
print('METADATA-LIKE OBJECTS:')
for x in metadata:print(x['size_bytes'],x['key'])
