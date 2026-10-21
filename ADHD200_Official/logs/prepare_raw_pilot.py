import csv,json,os,urllib.parse,urllib.request,collections,concurrent.futures
from pathlib import Path
W=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official')
P=W/'experiments/raw_pilot_18'; bucket='https://fcp-indi.s3.amazonaws.com/'
def read(path):
 with (W/path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
priority=read('metadata/raw_processing_priority.csv'); avail=read('metadata/raw_fmri_availability.csv'); objects=read('metadata/raw_bids_object_inventory.csv'); shared=read('metadata/raw_bids_site_metadata_inventory.csv')
features=Path('/Users/apple/Documents/Codex/2026-09-29/down/NeuroADHD/data/adhd200/features/cc200/cc200_connectivity_features.csv')
with features.open(newline='',encoding='utf-8') as f:existing={r['subject_id'].zfill(7) for r in csv.DictReader(f)}
# Priority is already readiness/representation ranked. Choose one ADHD and one control by rank per site; WashU has no ADHD-labeled candidate, so use two controls.
chosen=[]; sites=['KKI','NYU','NeuroIMAGE','OHSU','Peking_1','Peking_2','Peking_3','Pittsburgh','WashU']
for site in sites:
 candidates=[x for x in priority if x['site']==site and x['availability_status']=='READY_FOR_PREPROCESSING' and x['subject_id'].zfill(7) not in existing]
 adhd=[x for x in candidates if x['diagnosis_group']=='ADHD']; control=[x for x in candidates if x['diagnosis_group']=='Control']
 if len(adhd)>=1 and len(control)>=1:
  pair=[adhd[0],control[0]]
 elif len(control)>=2:
  pair=control[:2]
 elif len(adhd)>=2:
  pair=adhd[:2]
 else:raise RuntimeError(f'cannot select two subjects at {site}: ADHD={len(adhd)}, control={len(control)}')
 for n,x in enumerate(pair):
  reason='Top-ranked eligible ADHD candidate for this site; not present in existing 153-subject feature table.' if x['diagnosis_group']=='ADHD' else 'Top-ranked eligible Typically Developing Control candidate for this site; not present in existing 153-subject feature table.'
  if site=='WashU':reason='WashU has no diagnosis-valid ADHD candidate in the eligible raw cohort; selected one of its top-ranked controls to preserve two-subject site representation.'
  chosen.append({**x,'age':'','sex':'','selection_reason':reason})
# Attach age/sex from raw availability table (does not use external mapping).
av={x['subject_id'].zfill(7):x for x in avail}
for x in chosen:
 x['age']=av[x['subject_id'].zfill(7)]['age'];x['sex']=av[x['subject_id'].zfill(7)]['sex']
if len(chosen)!=18 or len({x['site'] for x in chosen})!=9:raise RuntimeError('selection failed exact 18 / 9 sites')
# map image objects and select most-specific applicable inherited sidecar JSON.
by=collections.defaultdict(list)
for o in objects:by[(o['site'],o['subject_id'].zfill(7))].append(o)
shared_by=collections.defaultdict(list)
for x in shared:shared_by[x['site']].append(x)
def ent_suffix(filename):
 n=filename
 for ext in ('.nii.gz','.nii','.json'):
  if n.lower().endswith(ext):n=n[:-len(ext)];break
 p=n.split('_');return p[:-1],p[-1]
def paired_sidecar(site, sid, image):
 ents,suf=ent_suffix(image); c=[]
 for s in shared_by[site]:
  if s['object_role']!='inherited_json_sidecar':continue
  e,ss=ent_suffix(s['filename'])
  if ss.lower()==suf.lower() and all(t in ents for t in e):c.append((len(e),s['relative_s3_path']))
 # Subject-level sidecars if present.
 for o in by[(site,sid)]:
  if o['extension'].lower()=='.json':
   e,ss=ent_suffix(o['filename'])
   if ss.lower()==suf.lower() and all(t in ents for t in e):c.append((len(e)+100,o['relative_s3_path']))
 if not c:raise RuntimeError(f'missing inherited JSON for {site}/{image}')
 return sorted(c,reverse=True)[0][1]
selected_keys=set(); required_type={}
for x in chosen:
 sid=x['subject_id'].zfill(7); site=x['site']; subj=by[(site,sid)]
 bold=sorted([o for o in subj if o['object_role']=='bold' and '_task-rest' in o['filename']],key=lambda z:z['filename'])
 anat=[o for o in subj if o['object_role']=='anat']
 if len(bold)<1 or len(anat)<1:raise RuntimeError(f'unexpected required-file multiplicity {site}/{sid}: bold {len(bold)} anat {len(anat)}')
 x['rs_fmri_s3_path']='|'.join('s3://fcp-indi/'+o['relative_s3_path'] for o in bold);x['anatomical_s3_path']='|'.join('s3://fcp-indi/'+o['relative_s3_path'] for o in anat)
 x['bold_key']='|'.join(o['relative_s3_path'] for o in bold);x['anat_key']='|'.join(o['relative_s3_path'] for o in anat)
 for o,typ in [(z,'rest_bold_nifti') for z in bold]+[(z,'anatomical_nifti') for z in anat]:
  selected_keys.add(o['relative_s3_path']);required_type[o['relative_s3_path']]=typ
  sk=paired_sidecar(site,sid,o['filename']);selected_keys.add(sk);required_type[sk]='bold_json_sidecar' if typ=='rest_bold_nifti' else 'anatomical_json_sidecar'
 # Required site root BIDS descriptors and participants file.
 for s in shared_by[site]:
  if s['filename'] in ('participants.tsv','dataset_description.json'):
   selected_keys.add(s['relative_s3_path']);required_type[s['relative_s3_path']]='required_site_bids_metadata'
 # file paths in manifest currently from inventory values.
manifest=[]
for x in chosen:
 manifest.append({k:x[k] for k in ['subject_id','site','diagnosis','age','sex','rs_fmri_s3_path','anatomical_s3_path','selection_reason']})
with (P/'pilot_subject_manifest.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(manifest[0]));w.writeheader();w.writerows(manifest)
# independently query official S3 HEAD object metadata for every planned file (no GET / no downloads).
def head(key):
 req=urllib.request.Request(bucket+urllib.parse.quote(key,safe='/'),method='HEAD')
 with urllib.request.urlopen(req,timeout=45) as r:return key,int(r.headers.get('Content-Length','0')),r.headers.get('ETag','').strip('"'),r.headers.get('Last-Modified','')
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as ex:heads=list(ex.map(head,sorted(selected_keys)))
headmap={k:(n,e,l) for k,n,e,l in heads}
for key in selected_keys:
 if headmap[key][0]<=0:raise RuntimeError(f'object absent/zero size on S3: {key}')
inv={o['relative_s3_path']:o for o in objects}
shinv={o['relative_s3_path']:o for o in shared}
files=[]
for key in sorted(selected_keys):
 if key in inv: site=inv[key]['site'];size=int(inv[key]['size_bytes'])
 else:site=shinv[key]['site'];size=int(shinv[key]['size_bytes'])
 hsize,etag,modified=headmap[key]
 if size!=hsize:raise RuntimeError(f'size disagreement for {key}: inventory={size} HEAD={hsize}')
 files.append({'s3_key':key,'site':site,'file_type':required_type[key],'expected_size_bytes':size,'etag':etag,'last_modified':modified,'local_relative_path':'BIDS/'+key.split('data/Projects/ADHD200/RawDataBIDS/',1)[1]})
N=sum(f['expected_size_bytes'] for f in files)
d={'selected_subjects':len(chosen),'sites':len(sites),'total_files':len(files),'total_bytes':N,'total_GB':N/1e9,'total_GiB':N/(1024**3),'size_source':'Fresh official S3 HEAD Content-Length for every selected required object; cross-checked with prior S3 listing metadata.','required_file_types':dict(collections.Counter(x['file_type'] for x in files)),'files':files,'subjects_by_site':{s:sum(x['site']==s for x in chosen) for s in sites},'diagnosis_counts':dict(collections.Counter(x['diagnosis'] for x in chosen))}
with (P/'qc/download_size.json').open('w') as f:json.dump(d,f,indent=2)
print(json.dumps({k:d[k] for k in ('selected_subjects','sites','total_files','total_bytes','total_GB','total_GiB','required_file_types','diagnosis_counts')},indent=2))
print('\nSelected:')
for x in chosen:print(x['site'],x['subject_id'],x['diagnosis'],x['age'],x['sex'],x['selection_reason'])
print('\nDOWNLOAD NOT STARTED; size manifest is saved.')
