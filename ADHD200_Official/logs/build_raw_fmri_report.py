import csv,json,os,collections,html
from pathlib import Path
W=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official')
def read(p):
 with (W/p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
rows=read('metadata/raw_fmri_availability.csv'); objs=read('metadata/raw_bids_object_inventory.csv'); shared=read('metadata/raw_bids_site_metadata_inventory.csv')
# Candidate denominator is the verified missing-compatible-derivative set in the previous phenotype match plus 8191384.
match=read('metadata/official_cc200_phenotype_match.csv')
cohort=[x for x in match if x['match_status']=='phenotype_without_CC200_derivative' and x['diagnosis_status']=='valid']
cohort += [x for x in match if x['normalized_subject_id']=='8191384' and x['diagnosis_status']=='valid']
manifest_n=len({(x['site'],x['normalized_subject_id']) for x in cohort})
ready=[x for x in rows if x['availability_status']=='READY_FOR_PREPROCESSING']
missingrs=[x for x in rows if x['availability_status']=='MISSING_RS_FMRI']
def isadhd(x):return x['diagnosis']!='Typically Developing Children'
adhd=sum(isadhd(x) for x in ready); ctrl=len(ready)-adhd
sites=sorted({x['site'] for x in rows})
# Exact shared metadata needed for sites represented among ready candidates, plus applicable inherited sidecars.
ready_sites={x['site'] for x in ready}; shared_by_site=collections.defaultdict(list)
for x in shared:shared_by_site[x['site']].append(x)
shared_required=set(); sidecar_names=set()
def split(k):
 name=os.path.basename(k)
 if name.endswith('.json'):name=name[:-5]
 p=name.split('_');return p[:-1],p[-1]
for x in ready:
 for o in [o for o in objs if o['site']==x['site'] and o['subject_id']==x['subject_id'] and o['object_role'] in ('bold','anat')]:
  ents,suffix=split(o['filename'])
  for m in shared_by_site[x['site']]:
   if m['object_role']!='inherited_json_sidecar':continue
   e,s=split(m['filename'])
   if s.lower()==suffix.lower() and all(t in ents for t in e):sidecar_names.add(m['relative_s3_path'])
for site in ready_sites:
 for m in shared_by_site[site]:
  if m['filename'] in ('participants.tsv','dataset_description.json'):shared_required.add(m['relative_s3_path'])
shared_req=shared_required|sidecar_names
shared_size=sum(int(x['size_bytes']) for x in shared if x['relative_s3_path'] in shared_req)
# Candidate download size for complete (ready) subjects; these are exact listed BOLD/T1 objects + their subject JSONs + deduplicated site-level metadata.
ready_ids={(x['site'],x['subject_id']) for x in ready}
ready_obj=[x for x in objs if (x['site'],x['subject_id']) in ready_ids]
func_bytes=sum(int(x['size_bytes']) for x in ready_obj if x['object_role']=='bold')
anat_bytes=sum(int(x['size_bytes']) for x in ready_obj if x['object_role']=='anat')
subject_meta=sum(int(x['size_bytes']) for x in ready_obj if x['object_role'] not in ('bold','anat'))
total=func_bytes+anat_bytes+subject_meta+shared_size
# Site summary.
site_rows=[]
for s in sites:
 q=[x for x in rows if x['site']==s]
 site_rows.append({'site':s,'total_missing_cc200_subjects':len(q),'rs_fmri_available':sum(x['resting_state_fmri_available']=='YES' for x in q),'anatomical_mri_available':sum(x['anatomical_mri_available']=='YES' for x in q),'both_available':sum(x['resting_state_fmri_available']=='YES' and x['anatomical_mri_available']=='YES' for x in q),'incomplete':sum(x['availability_status'] in ('INCOMPLETE','MISSING_METADATA','MISSING_RS_FMRI','MISSING_ANATOMICAL') for x in q),'unknown':sum(x['availability_status']=='UNKNOWN' for x in q),'ready_for_preprocessing':sum(x['availability_status']=='READY_FOR_PREPROCESSING' for x in q)})
with (W/'metadata/raw_fmri_site_summary.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(site_rows[0]));w.writeheader();w.writerows(site_rows)
# Deterministic greedy priority: readiness tiers first; among ready candidates select currently underrepresented diagnosis/site, balancing labels and spreading sites.
remaining=list(rows); picks=[]; bysite=collections.Counter(); byclass=collections.Counter()
for rank in range(1,len(remaining)+1):
 def score(x):
  stat=0 if x['availability_status']=='READY_FOR_PREPROCESSING' else 1
  cl='ADHD' if isadhd(x) else 'Control'
  return (stat,bysite[x['site']],byclass[cl],x['site'],x['subject_id'])
 x=min(remaining,key=score);remaining.remove(x); cl='ADHD' if isadhd(x) else 'Control'; bysite[x['site']]+=1;byclass[cl]+=1
 picks.append({'priority_rank':rank,'subject_id':x['subject_id'],'site':x['site'],'diagnosis':x['diagnosis'],'diagnosis_group':cl,'availability_status':x['availability_status'],'resting_state_fmri_available':x['resting_state_fmri_available'],'anatomical_mri_available':x['anatomical_mri_available'],'bids_metadata_available':x['bids_metadata_available'],'priority_basis':'Readiness tier, then greedy balance across diagnosis groups and sites; no model performance used.'})
with (W/'metadata/raw_processing_priority.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(picks[0]));w.writeheader();w.writerows(picks)
site_counts={s:sum(x['site']==s for x in ready) for s in sites}
tr=collections.Counter()
for x in ready:
 for val in x['rest_tr_seconds'].split('|'):
  if val:tr[val]+=1
# Site cohort source details and conservative compatibility caveats.
summary={'candidate_definition':'valid-diagnosis rows classified phenotype_without_CC200_derivative plus valid-diagnosis 8191384 with no compatible CC200 variant','missing_compatible_cc200_subjects':manifest_n,'listed_subject_prefixes':len(rows),'raw_s3_listing_failures':0,'rest_bold_available':sum(x['resting_state_fmri_available']=='YES' for x in rows),'anatomical_available':sum(x['anatomical_mri_available']=='YES' for x in rows),'both_available':sum(x['resting_state_fmri_available']=='YES' and x['anatomical_mri_available']=='YES' for x in rows),'potentially_processable':len(ready),'potential_adhd':adhd,'potential_control':ctrl,'potential_by_site':{s:n for s,n in site_counts.items() if n},'diagnosis_counts_potential':dict(collections.Counter(x['diagnosis'] for x in ready)),'tr_seconds_subject_counts':dict(tr),'potential_final_cohort_if_all_pass':153+len(ready),'potential_final_adhd200_only_addition':len(ready),'current_cohort':153,'download_size_bytes':{'functional_bold':func_bytes,'anatomical':anat_bytes,'subject_level_metadata':subject_meta,'deduplicated_shared_site_metadata_and_inherited_json':shared_size,'total':total},'download_size_gib':total/(1024**3),'candidate_subject_prefix_object_bytes_all_781':sum(int(x['raw_size_bytes']) for x in rows),'shared_metadata_object_count':len(shared_req),'shared_metadata_keys':sorted(shared_req),'NIfTI_payload_or_headers_fetched':False,'preprocessing_done':False,'model_training_done':False,'runtime':'Runtime not yet benchmarked.','expected_full_cpac_output_size':'Unknown; cannot derive exact output size from raw object listings.','machine':'8 GiB RAM, 8 CPU cores, approximately 341 GiB available at audit time; full C-PAC feasibility is uncertain and not benchmarked.','tr_counts':dict(tr),'missing_rest_subjects':[{'site':x['site'],'subject_id':x['subject_id'],'diagnosis':x['diagnosis']} for x in missingrs]}
with (W/'metadata/raw_fmri_summary.json').open('w') as f:json.dump(summary,f,indent=2)
# report
fmt=lambda b:f'{b:,} bytes ({b/1024**3:.2f} GiB)'
lines=['# ADHD-200 raw-fMRI availability and feasibility audit','',f"Audit date: 2026-10-05. Candidate denominator reconstructed from the official phenotype/CC200 match manifest: **{manifest_n} diagnosis-valid subjects** lacking a compatible CC200 derivative (780 without derivative plus 8191384 without a compatible variant).",'', '## Findings','',f'- S3 subject-prefix listings completed: **{len(rows)}/{manifest_n}**; listing failures: **0**.','- Resting-state BOLD listed: **{summary["rest_bold_available"]}**.','- Anatomical T1w/MPRAGE listed: **{summary["anatomical_available"]}**.','- Both listed: **{summary["both_available"]}**.','- Potentially processable (rest BOLD + anatomy + applicable sidecars/TR + site-level BIDS metadata): **{len(ready)}**. This is an availability classification, not a QC pass.','- Potentially processable diagnosis groups: **%d ADHD** and **%d Typically Developing Controls**.'%(adhd,ctrl),'- Sites represented among potentially processable subjects: '+', '.join(f'{s} ({n})' for s,n in site_counts.items() if n)+'.',f'- Potential final cohort if every candidate subsequently passes preprocessing and CC200 QC: **{153+len(ready)}** (153 existing + {len(ready)} potential additions). Do not call this a usable cohort yet.','', '## Site summary','', '| Site | Candidates | RS-fMRI | Anatomy | Both | Ready | Missing/incomplete | Unknown |','|---|---:|---:|---:|---:|---:|---:|---:|']
for x in site_rows:lines.append(f"| {x['site']} | {x['total_missing_cc200_subjects']} | {x['rs_fmri_available']} | {x['anatomical_mri_available']} | {x['both_available']} | {x['ready_for_preprocessing']} | {x['incomplete']} | {x['unknown']} |")
lines+=['','## Preprocessing compatibility','', 'Target requested: `pipeline_adhd200-benchmark__freq-filter`, 0.01–0.1 Hz, global signal regression OFF, CC200. The raw BIDS objects do not have prior C-PAC processing status, so all candidates require preprocessing; no raw image can establish the eventual derivative configuration match until processed.', '', 'Among the 775 potentially processable subjects, inherited BIDS metadata provided a positive rest task label and numeric TR for each listed rest BOLD. Observed TR values (subject counts): '+', '.join(f'{t} s: {n}' for t,n in sorted(tr.items(),key=lambda z:float(z[0])))+'. Acquisition/TR variation is present across sites and needs to be represented correctly in the C-PAC inputs/configuration.', '', 'No NIfTI data or headers were fetched. Therefore image dimensions, affine/orientation, payload integrity, missing volumes, and anatomical/BOLD alignment remain **unknown**. These are explicit preprocessing compatibility checks, not inferred from file presence. No event/task files were listed in the subject inventories; all BOLD matches are rest task. Six NYU subjects (five ADHD, one control) have T1w but no rest BOLD listed and are excluded from potentially processable count.']
lines+=['','## Exact listed raw download size for the potentially processable set','',f'- Functional rest BOLD: {fmt(func_bytes)}',f'- Anatomical T1w/MPRAGE: {fmt(anat_bytes)}',f'- Subject-level metadata: {fmt(subject_meta)}',f'- Deduplicated shared site metadata and inherited JSON sidecars: {fmt(shared_size)} across {len(shared_req)} objects',f'- **Total: {fmt(total)}**', '', 'All values are exact S3 object `Size` metadata for the candidate subjects classified READY, plus the deduplicated site-root files required to interpret them. This covers 775 subjects only; it excludes the six without rest BOLD and does not download any object. The full subject-prefix inventory across all 781 sums to '+fmt(summary['candidate_subject_prefix_object_bytes_all_781'])+' (includes the six subjects’ available files).', '', '## Feasibility','', 'The 775-subject workload is a substantial multi-site C-PAC run. At audit time the machine had 8 GiB RAM, 8 CPU cores, and about 341 GiB free disk. The listed raw set fits within available disk by byte count, but that does not account for temporary processing data, work directories, or derivative outputs. Expected C-PAC output size is unknown from the available evidence. **Runtime not yet benchmarked.** The machine’s 8 GiB RAM makes running all candidates concurrently imprudent; processing feasibility should be checked on a small representative pilot after a cohort is selected.', '', '## Priority and limitations','', 'The file `metadata/raw_processing_priority.csv` ranks every subject. It puts READY subjects first, then uses deterministic greedy balancing across ADHD/control and site representation. It does not use model performance. This is a planning list, not a decision that all 775 should be processed.', '', 'The status READY_FOR_PREPROCESSING means S3 lists rest BOLD and T1w objects, their applicable JSON sidecars and valid TR/task metadata, and site participants/dataset metadata. It does not certify NIfTI readability, image dimensions, motion/QC, successful preprocessing, or CC200 quality. No raw MRI GET, preprocessing, feature extraction, or model training was performed. The original NeuroADHD project and 153-subject feature table were not modified.']
(W/'reports/raw_fmri_availability_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
body='\n'.join('<p>'+html.escape(x)+'</p>' for x in lines if x and not x.startswith('|'))
table='<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in site_rows[0])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in x.values())+'</tr>' for x in site_rows)+'</tbody></table>'
(W/'reports/raw_fmri_availability_report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>ADHD-200 raw-fMRI availability audit</title><style>body{font:16px system-ui;max-width:1000px;margin:3rem auto;padding:0 1rem;line-height:1.55}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.4rem .7rem;text-align:left}</style></head><body><h1>ADHD-200 raw-fMRI availability and feasibility audit</h1>'+body+'<h2>Site summary</h2>'+table+'</body></html>',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('missing_compatible_cc200_subjects','rest_bold_available','anatomical_available','both_available','potentially_processable','potential_adhd','potential_control','potential_by_site','potential_final_cohort_if_all_pass','download_size_bytes')},indent=2))
