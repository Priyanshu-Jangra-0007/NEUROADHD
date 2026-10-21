from __future__ import annotations
import csv, json, os, re, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET, concurrent.futures, collections, statistics
from pathlib import Path

ROOT=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official')
BUCKET='https://fcp-indi.s3.amazonaws.com/'
BIDS_ROOT='data/Projects/ADHD200/RawDataBIDS/'
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}

# Candidate cohort is reconstructed from the previous phenotype/CC200 manifest, not hard-coded.
with (ROOT/'metadata/official_cc200_phenotype_match.csv').open(newline='',encoding='utf-8') as f:
    matched=list(csv.DictReader(f))
candidates=[r for r in matched if r['match_status']=='phenotype_without_CC200_derivative' and r['diagnosis_status']=='valid']
# One diagnosis-valid CC200 subject was subsequently established to have no compatible variant.
for r in matched:
    if r['normalized_subject_id']=='8191384' and r['diagnosis_status']=='valid':
        candidates.append(r)
# stable unique (site, subject), retaining source phenotype fields
candidate_by_key={(r['site'],r['normalized_subject_id']):r for r in candidates}
candidates=sorted(candidate_by_key.values(),key=lambda r:(r['site'],r['normalized_subject_id']))
print('candidate_count_from_prior_audit',len(candidates),flush=True)

def list_objects(prefix:str)->list[dict]:
    out=[]; token=None
    while True:
        q={'list-type':'2','prefix':prefix,'max-keys':'1000'}
        if token:q['continuation-token']=token
        url=BUCKET+'?'+urllib.parse.urlencode(q)
        last=None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url,timeout=45) as resp: data=resp.read()
                x=ET.fromstring(data); break
            except Exception as e:
                last=e; time.sleep(.5*(attempt+1))
        else: raise RuntimeError(f'{prefix}: {last}')
        for e in x.findall('s:Contents',NS):
            key=e.findtext('s:Key',namespaces=NS)
            if key:
                out.append({'key':key,'size_bytes':int(e.findtext('s:Size',default='0',namespaces=NS) or 0)})
        if x.findtext('s:IsTruncated',namespaces=NS)!='true':break
        token=x.findtext('s:NextContinuationToken',namespaces=NS)
        if not token:raise RuntimeError('truncated S3 response without continuation token')
    return out

def candidate_prefix(r):
    sid=r['normalized_subject_id'].zfill(7)
    return BIDS_ROOT+r['site']+'/sub-'+sid+'/'

# Read only site-root BIDS metadata directory entries (delimiter listing, no object bodies).
sites=sorted({r['site'] for r in candidates})
def site_root(site):
    prefix=BIDS_ROOT+site+'/'
    q={'list-type':'2','prefix':prefix,'delimiter':'/','max-keys':'1000'}
    with urllib.request.urlopen(BUCKET+'?'+urllib.parse.urlencode(q),timeout=45) as resp: x=ET.fromstring(resp.read())
    return [{'key':e.findtext('s:Key',namespaces=NS),'size_bytes':int(e.findtext('s:Size',default='0',namespaces=NS) or 0)} for e in x.findall('s:Contents',NS)]
site_root_objects={}
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
    for site,objs in zip(sites,ex.map(site_root,sites)):site_root_objects[site]=objs
print('site_roots_read',len(site_root_objects),flush=True)

# List object metadata for each individual subject prefix. No NIfTI bodies are fetched.
obj_by_key={}; failed={}
def fetch_subject(r):
    return (r, list_objects(candidate_prefix(r)))
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
    futs={ex.submit(fetch_subject,r):r for r in candidates}
    done=0
    for fut in concurrent.futures.as_completed(futs):
        r=futs[fut]; k=(r['site'],r['normalized_subject_id'])
        try:
            _,objs=fut.result(); obj_by_key[k]=objs
        except Exception as e:failed[k]=str(e)
        done+=1
        if done%100==0:print('subject_prefixes_listed',done,'/',len(candidates),flush=True)
print('subject_prefix_listing_complete',len(obj_by_key),'failed',len(failed),flush=True)

# Download only small BIDS JSON sidecars into memory to read protocol fields such as RepetitionTime.
# No NIfTI, MRI, or event contents are fetched.
def is_img_nii(k):return k.lower().endswith(('.nii','.nii.gz'))
def modality(k):
  name=k.rsplit('/',1)[-1]
  if '/func/' in k and re.search(r'_bold\.nii(?:\.gz)?$',name,re.I):return 'bold'
  if '/anat/' in k and re.search(r'_(?:t1w|mprage)\.nii(?:\.gz)?$',name,re.I):return 'anat'
  return ''
json_targets=[]
for key,objs in obj_by_key.items():
    keys={o['key'] for o in objs}
    for o in objs:
        m=modality(o['key'])
        if not m:continue
        stem=re.sub(r'\.nii(?:\.gz)?$','',o['key'],flags=re.I)
        j=stem+'.json'
        if j in keys:json_targets.append(j)

def get_json(key):
    req=urllib.request.Request(BUCKET+urllib.parse.quote(key,safe='/'))
    with urllib.request.urlopen(req,timeout=30) as resp:return key,resp.read()
sidecars={};json_failed={}
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:
    futs={ex.submit(get_json,k):k for k in sorted(set(json_targets))}
    done=0
    for fut in concurrent.futures.as_completed(futs):
        k=futs[fut]
        try:
            key,b=fut.result();sidecars[key]=json.loads(b.decode('utf-8-sig'))
        except Exception as e:json_failed[k]=str(e)
        done+=1
        if done%200==0:print('sidecars_read',done,'/',len(futs),flush=True)
print('sidecars_complete',len(sidecars),'failed',len(json_failed),flush=True)

# Build per-subject object rows and subject-level availability.
all_objects=[]; rows=[]
expected_labels={'Typically Developing Children','ADHD-Combined','ADHD-Hyperactive/Impulsive','ADHD-Inattentive'}
for r in candidates:
    key=(r['site'],r['normalized_subject_id']); sid=r['normalized_subject_id'].zfill(7); prefix=candidate_prefix(r)
    if key in failed:
        rows.append({'subject_id':sid,'site':r['site'],'diagnosis':r['diagnosis_raw'],'age':r.get('age_years',''),'sex':r.get('sex',''),'resting_state_fmri_available':'UNKNOWN','anatomical_mri_available':'UNKNOWN','bids_metadata_available':'UNKNOWN','functional_file_count':'','anatomical_file_count':'','raw_size_bytes':'','availability_status':'UNKNOWN','reason':'S3 subject-prefix listing failed: '+failed[key]})
        continue
    objs=obj_by_key.get(key,[]); keys={o['key'] for o in objs}
    for o in objs:
        k=o['key']; name=k.rsplit('/',1)[-1]
        kind=modality(k)
        ext='.nii.gz' if k.lower().endswith('.nii.gz') else os.path.splitext(k)[1].lower()
        role=kind or ('functional_sidecar_json' if '/func/' in k and name.lower().endswith('.json') else 'anatomical_sidecar_json' if '/anat/' in k and name.lower().endswith('.json') else 'events' if re.search(r'_events\.(tsv|json)$',name,re.I) else 'session_metadata' if name.lower().endswith(('.tsv','.json')) else 'other')
        all_objects.append({'subject_id':sid,'site':r['site'],'object_role':role,'relative_s3_path':k,'filename':name,'extension':ext,'size_bytes':o['size_bytes']})
    bold=[o for o in objs if modality(o['key'])=='bold']
    anat=[o for o in objs if modality(o['key'])=='anat']
    # Any other functional NIfTI is counted in file count; RS availability only counts BIDS task-rest BOLD.
    functional_nii=[o for o in objs if '/func/' in o['key'] and is_img_nii(o['key'])]
    rest_bold=[o for o in bold if re.search(r'_task-rest(?:_|\.)',o['key'].rsplit('/',1)[-1],re.I)]
    func_json=[];anat_json=[];sidecar_issues=[];trs=[];acq=[];dims=[]
    for o in rest_bold:
        stem=re.sub(r'\.nii(?:\.gz)?$','',o['key'],flags=re.I);j=stem+'.json'
        if j not in keys:sidecar_issues.append(os.path.basename(stem)+': missing JSON')
        elif j not in sidecars:sidecar_issues.append(os.path.basename(j)+': JSON unavailable/unparsed')
        else:
            js=sidecars[j];func_json.append(j)
            tr=js.get('RepetitionTime')
            if isinstance(tr,(int,float)):trs.append(float(tr))
            else:sidecar_issues.append(os.path.basename(j)+': missing RepetitionTime')
            acq.append({k:js.get(k) for k in ('TaskName','RepetitionTime','EchoTime','Manufacturer','ManufacturersModelName','MagneticFieldStrength','PhaseEncodingDirection','TotalReadoutTime') if k in js})
            if 'ReconMatrixPE' in js:dims.append(js['ReconMatrixPE'])
    for o in anat:
        stem=re.sub(r'\.nii(?:\.gz)?$','',o['key'],flags=re.I);j=stem+'.json'
        if j not in keys:sidecar_issues.append(os.path.basename(stem)+': missing JSON')
        elif j not in sidecars:sidecar_issues.append(os.path.basename(j)+': JSON unavailable/unparsed')
        else:anat_json.append(j)
    shared=site_root_objects.get(r['site'],[])
    shared_names={os.path.basename(x['key']) for x in shared}
    shared_bids_meta=('participants.tsv' in shared_names and 'dataset_description.json' in shared_names)
    pairs_complete=(len(rest_bold)>0 and len(func_json)==len(rest_bold) and len(anat)>0 and len(anat_json)==len(anat))
    bids_meta=bool(shared_bids_meta and pairs_complete)
    data_total=sum(o['size_bytes'] for o in objs)
    rest_present=bool(rest_bold);anat_present=bool(anat)
    # On object availability only. NIfTI header/payload integrity is not tested.
    if not rest_present and not anat_present:status='MISSING_RS_FMRI';reason='No task-rest BOLD NIfTI and no T1w/MPRAGE NIfTI objects listed.'
    elif not rest_present:status='MISSING_RS_FMRI';reason='No task-rest BOLD NIfTI object listed.'
    elif not anat_present:status='MISSING_ANATOMICAL';reason='No T1w/MPRAGE NIfTI object listed.'
    elif not shared_bids_meta and not func_json and not anat_json:status='MISSING_METADATA';reason='Imaging listed but required sidecar/site BIDS metadata are absent.'
    elif not bids_meta or sidecar_issues:status='INCOMPLETE';reason='One or more image sidecars/TR or shared participants/dataset metadata are incomplete; '+('; '.join(sidecar_issues[:8]) if sidecar_issues else 'site-level participants.tsv/dataset_description.json missing')
    else:status='READY_FOR_PREPROCESSING';reason='Rest BOLD, T1w/MPRAGE, paired JSON sidecars with RepetitionTime, and site-level BIDS metadata listed. NIfTI payload/header integrity still unverified.'
    # Separate exact required-file size estimates: BOLD NIfTIs, T1 NIfTIs, sidecars/session/event/shared metadata, other.
    rows.append({'subject_id':sid,'site':r['site'],'diagnosis':r['diagnosis_raw'],'age':r.get('age_years',''),'sex':r.get('sex',''),'resting_state_fmri_available':'YES' if rest_present else 'NO','anatomical_mri_available':'YES' if anat_present else 'NO','bids_metadata_available':'YES' if bids_meta else 'NO','functional_file_count':len(functional_nii),'anatomical_file_count':len(anat),'raw_size_bytes':data_total,'availability_status':status,'reason':reason,'rest_bold_file_count':len(rest_bold),'rest_tr_seconds':'|'.join(map(str,sorted(set(trs)))),'rest_bold_names':'|'.join(os.path.basename(o['key']) for o in rest_bold),'anatomical_names':'|'.join(os.path.basename(o['key']) for o in anat),'sidecar_json_count':len(func_json)+len(anat_json),'events_file_count':sum(1 for o in objs if re.search(r'_events\.(tsv|json)$',o['key'],re.I)),'other_subject_object_count':sum(1 for o in objs if not modality(o['key']) and not o['key'].lower().endswith('.json') and not o['key'].lower().endswith('.tsv')),'functional_bytes':sum(o['size_bytes'] for o in rest_bold),'anatomical_bytes':sum(o['size_bytes'] for o in anat),'subject_metadata_bytes':sum(o['size_bytes'] for o in objs if o['key'].lower().endswith(('.json','.tsv'))),'other_bytes':sum(o['size_bytes'] for o in objs if not modality(o['key']) and not o['key'].lower().endswith(('.json','.tsv'))),'sidecar_issues':'|'.join(sidecar_issues),'rest_tr_unique_count':len(set(trs)),'rest_acquisition_metadata':json.dumps(acq,separators=(',',':')),'site_shared_metadata_available':'YES' if shared_bids_meta else 'NO','raw_nifti_dimensions':'NOT_INSPECTED_NO_NIFTI_GET'})

# Save subject table (specified required columns first; supplemental audit detail follows).
basecols=['subject_id','site','diagnosis','age','sex','resting_state_fmri_available','anatomical_mri_available','bids_metadata_available','functional_file_count','anatomical_file_count','raw_size_bytes','availability_status','reason']
extra=[x for x in (rows[0].keys() if rows else []) if x not in basecols]
with (ROOT/'metadata/raw_fmri_availability.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=basecols+extra);w.writeheader();w.writerows(rows)
with (ROOT/'metadata/raw_bids_object_inventory.csv').open('w',newline='',encoding='utf-8') as f:
    fields=['subject_id','site','object_role','relative_s3_path','filename','extension','size_bytes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(all_objects)
with (ROOT/'logs/raw_bids_listing_failures.json').open('w') as f:json.dump({'subject_listing_failures':{str(k):v for k,v in failed.items()},'sidecar_failures':json_failed,'subject_prefixes_completed':len(obj_by_key),'candidate_count':len(candidates),'sidecars_found':len(sidecars),'sidecars_expected':len(json_targets)},f,indent=2)
print('candidate_count',len(candidates),'subjects_with_objects',sum(bool(v) for v in obj_by_key.values()),'subject_failures',len(failed),'sidecar_json_targets',len(json_targets),'sidecar_get_failures',len(json_failed),'inventory_objects',len(all_objects),flush=True)
