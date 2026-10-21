import csv,json,os,re,urllib.request,urllib.parse,xml.etree.ElementTree as ET,concurrent.futures
from pathlib import Path
W=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official'); base='https://fcp-indi.s3.amazonaws.com/'; root='data/Projects/ADHD200/RawDataBIDS/'; ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
with (W/'metadata/raw_fmri_availability.csv').open(newline='') as f: rows=list(csv.DictReader(f))
with (W/'metadata/raw_bids_object_inventory.csv').open(newline='') as f: objects=list(csv.DictReader(f))
by=__import__('collections').defaultdict(list)
for o in objects:by[(o['site'],o['subject_id'])].append(o)
sites=sorted({r['site'] for r in rows})
def root_objects(site):
 p=root+site+'/'
 q=urllib.parse.urlencode({'list-type':'2','prefix':p,'delimiter':'/','max-keys':'1000'})
 with urllib.request.urlopen(base+'?'+q,timeout=45) as resp:x=ET.fromstring(resp.read())
 return [{'key':e.findtext('s:Key',namespaces=ns),'size_bytes':int(e.findtext('s:Size',default='0',namespaces=ns) or 0)} for e in x.findall('s:Contents',ns)]
roots={}
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
 for site,objs in zip(sites,ex.map(root_objects,sites)):roots[site]=objs
json_keys=[o['key'] for objs in roots.values() for o in objs if o['key'].lower().endswith('.json')]
# Subject-level JSON sidecars are tiny metadata objects; never fetch image payloads.
subject_json_keys=sorted({o['relative_s3_path'] for o in objects if o['extension'].lower()=='.json'})
json_keys=sorted(set(json_keys)|set(subject_json_keys))
def getjson(k):
 with urllib.request.urlopen(base+urllib.parse.quote(k,safe='/'),timeout=30) as resp:return k,json.loads(resp.read().decode('utf-8-sig'))
side={}
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as ex:
 for k,j in ex.map(getjson,json_keys):side[k]=j

def split_name(k):
 name=k.rsplit('/',1)[-1]
 if name.lower().endswith('.nii.gz'):name=name[:-7]
 elif name.lower().endswith('.nii'):name=name[:-4]
 elif name.lower().endswith('.json'):name=name[:-5]
 parts=name.split('_')
 return parts[:-1],parts[-1] if parts else ''
def match_sidecar(image_key,site,objects_for_subject):
 ents,suffix=split_name(image_key)
 candidates=[]
 # Subject/session-level JSONs and dataset/site-root JSONs may be inherited by BIDS rules.
 alljson=[]
 for o in objects_for_subject:
  if o['extension'].lower()=='.json':alljson.append((o['relative_s3_path'],int(o['size_bytes'])))
 for o in roots[site]:
  if o['key'].lower().endswith('.json'):alljson.append((o['key'],o['size_bytes']))
 for key,size in alljson:
  e,s=split_name(key)
  if s.lower()!=suffix.lower():continue
  if all(t in ents for t in e):candidates.append((len(e),key,size))
 if not candidates:return None
 candidates.sort(reverse=True)
 best=candidates[0]
 # Multiple same-specificity matching sidecars are ambiguous; require unique choice.
 if len(candidates)>1 and candidates[1][0]==best[0] and candidates[1][1]!=best[1]:return 'AMBIGUOUS'
 return best[1]
site_meta_rows=[]
for site,objs in roots.items():
 for o in objs:
  name=os.path.basename(o['key'])
  role='site_shared_metadata' if name in ('participants.tsv','dataset_description.json','README','README.txt') else 'inherited_json_sidecar' if name.endswith('.json') else 'site_other'
  site_meta_rows.append({'site':site,'relative_s3_path':o['key'],'filename':name,'size_bytes':o['size_bytes'],'object_role':role})
with (W/'metadata/raw_bids_site_metadata_inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['site','relative_s3_path','filename','size_bytes','object_role']);w.writeheader();w.writerows(site_meta_rows)
# Enrich existing availability rows with inherited metadata and true TaskName/TR info.
for r in rows:
 site=r['site'];sid=r['subject_id']; subj=by[(site,sid)]
 fs=[o for o in subj if o['object_role']=='bold']; ats=[o for o in subj if o['object_role']=='anat']
 sroots=roots[site]; sharednames={os.path.basename(o['key']) for o in sroots}
 shared_ok=('participants.tsv' in sharednames and 'dataset_description.json' in sharednames)
 bold_records=[]; anat_records=[]; issues=[]; rootkeys_used=set(); task_events=[]
 for o in fs:
  k=o['relative_s3_path'];sj=match_sidecar(k,site,subj)
  if sj is None or sj=='AMBIGUOUS':issues.append(os.path.basename(k)+': missing/ambiguous BOLD JSON');continue
  js=side[sj];rootkeys_used.add(sj)
  taskname=js.get('TaskName');tr=js.get('RepetitionTime')
  bold_records.append({'file':os.path.basename(k),'sidecar':os.path.basename(sj),'task_name':taskname,'tr':tr,'acq':next((x for x in os.path.basename(k).split('_') if x.startswith('acq-')),''),'manufacturer':js.get('Manufacturer',''),'model':js.get('ManufacturersModelName',''),'field_strength':js.get('MagneticFieldStrength','')})
  if not isinstance(tr,(float,int)) or tr<=0:issues.append(os.path.basename(k)+': missing/invalid RepetitionTime')
  if taskname is None:issues.append(os.path.basename(sj)+': missing TaskName')
  if re.search(r'_events\.(?:tsv|json)$',k,re.I):task_events.append(k)
 for o in ats:
  k=o['relative_s3_path'];sj=match_sidecar(k,site,subj)
  if sj is None or sj=='AMBIGUOUS':issues.append(os.path.basename(k)+': missing/ambiguous anatomical JSON');continue
  js=side[sj];rootkeys_used.add(sj)
  anat_records.append({'file':os.path.basename(k),'sidecar':os.path.basename(sj),'manufacturer':js.get('Manufacturer',''),'model':js.get('ManufacturersModelName',''),'field_strength':js.get('MagneticFieldStrength','')})
 rest=[x for x in bold_records if str(x['task_name']).lower()=='rest']
 all_t1_sided=len(anat_records)==len(ats) and len(ats)>0
 all_bold_sided=len(bold_records)==len(fs)
 metadata_ok=shared_ok and all_t1_sided and all_bold_sided and all(isinstance(x['tr'],(int,float)) and x['tr']>0 and x['task_name'] for x in bold_records)
 has_rs=bool(rest);has_anat=bool(ats)
 if not has_rs:status='MISSING_RS_FMRI';reason='No functional BOLD image with inherited/paired TaskName=rest metadata.'
 elif not has_anat:status='MISSING_ANATOMICAL';reason='No T1w/MPRAGE anatomical NIfTI object listed.'
 elif not shared_ok and not (rootkeys_used):status='MISSING_METADATA';reason='Required BIDS site metadata and applicable sidecars are not available.'
 elif metadata_ok:status='READY_FOR_PREPROCESSING';reason=f"{len(rest)} resting BOLD file(s), {len(ats)} anatomical file(s); BIDS inherited sidecars and RepetitionTime available (TR: {','.join(map(str,sorted({x['tr'] for x in rest}))) } s). NIfTI payload/header integrity not inspected."
 else:status='INCOMPLETE';reason='BIDS image or shared metadata is incomplete: '+('; '.join(issues[:6]) if issues else 'missing site participants.tsv/dataset_description.json')
 r['resting_state_fmri_available']='YES' if has_rs else 'NO'
 r['anatomical_mri_available']='YES' if has_anat else 'NO'
 r['bids_metadata_available']='YES' if metadata_ok else 'NO'
 r['availability_status']=status;r['reason']=reason
 r['rest_bold_run_count']=len(rest);r['rest_tr_seconds']='|'.join(map(str,sorted({x['tr'] for x in rest})))
 r['rest_bold_names']='|'.join(x['file'] for x in rest);r['anatomical_names']='|'.join(x['file'] for x in anat_records)
 r['sidecar_json_count']=len(rootkeys_used);r['site_shared_metadata_available']='YES' if shared_ok else 'NO'
 r['rest_acquisition_metadata']=json.dumps(rest,separators=(',',':'))
 r['anatomical_acquisition_metadata']=json.dumps(anat_records,separators=(',',':'))
 r['sidecar_issues']='|'.join(issues)
 r['events_file_count']=sum(1 for o in subj if re.search(r'_events\.(?:tsv|json)$',o['filename'],re.I))
# Rewrite table, preserving fields and desired first columns.
basecols=['subject_id','site','diagnosis','age','sex','resting_state_fmri_available','anatomical_mri_available','bids_metadata_available','functional_file_count','anatomical_file_count','raw_size_bytes','availability_status','reason']
fields=list(dict.fromkeys(basecols+[k for r in rows for k in r]))
with (W/'metadata/raw_fmri_availability.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (W/'metadata/site_bids_metadata_summary.json').open('w') as f:
 json.dump({'sources':'S3 object listing plus GET of small BIDS JSON sidecars only; no NIfTI/raw MRI fetched.','sites':{s:{'root_objects':len(v),'participants_tsv':'participants.tsv' in {os.path.basename(o['key']) for o in v},'dataset_description':'dataset_description.json' in {os.path.basename(o['key']) for o in v},'json_sidecars':[{ 'key':k,'size_bytes':next((o['size_bytes'] for o in v if o['key']==k),next((int(o['size_bytes']) for o in objects if o['relative_s3_path']==k),0)),'fields':{name:side[k].get(name) for name in ('TaskName','RepetitionTime','EchoTime','Manufacturer','ManufacturersModelName','MagneticFieldStrength','PhaseEncodingDirection') if name in side[k]}} for k in side if k.startswith(root+s+'/')] } for s,v in roots.items()},'subject_level_json_keys':subject_json_keys},f,indent=2)
print('updated_rows',len(rows),'statuses',__import__('collections').Counter(r['availability_status'] for r in rows),'RS',sum(r['resting_state_fmri_available']=='YES' for r in rows),'anat',sum(r['anatomical_mri_available']=='YES' for r in rows),'metadata_yes',sum(r['bids_metadata_available']=='YES' for r in rows))
