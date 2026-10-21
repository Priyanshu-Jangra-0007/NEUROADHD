import csv,json,gzip,struct,math,os
from pathlib import Path
W=Path('/Users/apple/Documents/Codex/2026-09-29/down/ADHD200_Official');P=W/'experiments/raw_pilot_18'
size=json.load(open(P/'qc/download_size.json')); manifest=list(csv.DictReader(open(P/'pilot_subject_manifest.csv'))); dl=list(csv.DictReader(open(P/'qc/download_manifest.csv')))
files={x['s3_key']:x for x in dl}; selected={x['s3_key']:x for x in size['files']}
def path(k):return P/'raw'/selected[k]['local_relative_path']
def nifti(k):
 p=path(k); n=p.stat().st_size
 if n<=0:raise ValueError('zero length')
 with gzip.open(p,'rb') as g: hdr=g.read(352)
 if len(hdr)<348:raise ValueError(f'truncated header ({len(hdr)} bytes)')
 szle=struct.unpack('<i',hdr[:4])[0];szen=struct.unpack('>i',hdr[:4])[0]
 endian='<' if szle==348 else '>' if szen==348 else None
 if endian is None:raise ValueError(f'not a NIfTI-1 header: sizeof_hdr LE={szle} BE={szen}')
 dims=struct.unpack(endian+'8h',hdr[40:56]);nd=int(dims[0]);shape=tuple(int(v) for v in dims[1:nd+1])
 pix=struct.unpack(endian+'8f',hdr[76:108]);vox=struct.unpack(endian+'f',hdr[108:112])[0]
 datatype=struct.unpack(endian+'h',hdr[70:72])[0];bitpix=struct.unpack(endian+'h',hdr[72:74])[0]
 if not (1<=nd<=7) or len(shape)!=nd or any(v<=0 for v in shape):raise ValueError(f'invalid dimensions dim={dims}')
 if not all(math.isfinite(x) and x>0 for x in pix[1:nd+1]):raise ValueError(f'invalid voxel sizes pixdim={pix}')
 # Read through entire gzip member; this verifies decompression and its gzip CRC/trailer.
 decompressed=352
 with gzip.open(p,'rb') as g:
  g.read(352)
  while True:
   b=g.read(4*1024*1024)
   if not b:break
   decompressed+=len(b)
 if datatype==0 or bitpix<=0:raise ValueError(f'invalid datatype/bitpix {datatype}/{bitpix}')
 required=int(vox)+math.prod(shape)*bitpix//8
 if decompressed<required:raise ValueError(f'truncated data: decompressed {decompressed}, expected at least {required}')
 return {'shape':shape,'dim_count':nd,'voxel_size_mm':tuple(round(float(x),6) for x in pix[1:nd+1]),'TR_seconds':float(pix[4]) if nd>=4 else '', 'header_readable':True,'gzip_integrity':'PASS','decompressed_bytes':decompressed,'datatype':datatype,'bitpix':bitpix,'required_bytes_min':required}
# find applicable sidecar from the exact downloaded file set using BIDS entity inheritance.
def parts(filename):
 n=filename
 for ext in ('.nii.gz','.nii','.json'):
  if n.lower().endswith(ext):n=n[:-len(ext)];break
 a=n.split('_');return a[:-1],a[-1]
root='data/Projects/ADHD200/RawDataBIDS/'
def sidecar_for(k):
 ent,suf=parts(os.path.basename(k)); cand=[]
 for q in selected:
  if not q.lower().endswith('.json'):continue
  # Restrict to same site root or same subject directory.
  if not (q.startswith(k[:k.index('/sub-')]+'/') if '/sub-' in k else q.startswith(k.rsplit('/',1)[0]+'/')):continue
  qe,qs=parts(os.path.basename(q))
  if qs.lower()==suf.lower() and all(t in ent for t in qe):cand.append((len(qe),q))
 if not cand:raise ValueError('no downloaded applicable sidecar')
 return sorted(cand,reverse=True)[0][1]
results=[]
for subj in manifest:
 sid=subj['subject_id'];site=subj['site'];boldkeys=[f['s3_key'] for f in size['files'] if f['site']==site and f['file_type']=='rest_bold_nifti' and f'/sub-{sid}/' in f['s3_key']];anatkeys=[f['s3_key'] for f in size['files'] if f['site']==site and f['file_type']=='anatomical_nifti' and f'/sub-{sid}/' in f['s3_key']]
 issues=[];bs=[];ats=[];trs=[];bjsonvalid=True;ajsonvalid=True
 for k in boldkeys:
  try:
   n=nifti(k);bs.append(n)
   sj=sidecar_for(k);js=json.loads(path(sj).read_text(encoding='utf-8-sig'))
   tr=js.get('RepetitionTime');task=js.get('TaskName')
   if not isinstance(tr,(int,float)) or tr<=0:raise ValueError(f'BOLD JSON missing positive RepetitionTime: {sj}')
   if not isinstance(task,str) or task.lower()!='rest':raise ValueError(f'BOLD JSON TaskName not rest: {sj} TaskName={task!r}')
   trs.append(float(tr))
   if len(n['shape'])<4:raise ValueError(f'BOLD image is not 4D: {k} {n["shape"]}')
  except Exception as e:bjsonvalid=False;issues.append(f'BOLD {os.path.basename(k)}: {type(e).__name__}: {e}')
 for k in anatkeys:
  try:
   n=nifti(k);ats.append(n);sj=sidecar_for(k);js=json.loads(path(sj).read_text(encoding='utf-8-sig'))
   if not isinstance(js,dict) or not js:raise ValueError(f'anatomical JSON empty or not an object: {sj}')
   if len(n['shape']) not in (3,4) or (len(n['shape'])==4 and n['shape'][3]!=1):raise ValueError(f'anatomical image is not 3D/ singleton 4D: {k} {n["shape"]}')
  except Exception as e:ajsonvalid=False;issues.append(f'ANAT {os.path.basename(k)}: {type(e).__name__}: {e}')
 # Verify all downloaded JSON (including inherited sidecars and shared BIDS metadata) parse.
 json_keys=[f['s3_key'] for f in size['files'] if f['site']==site and f['file_type'].endswith('json_sidecar')]
 for k in json_keys:
  try:
   obj=json.loads(path(k).read_text(encoding='utf-8-sig'))
   if not isinstance(obj,dict) or not obj:raise ValueError('empty JSON object')
  except Exception as e:issues.append(f'JSON {os.path.basename(k)}: {type(e).__name__}: {e}')
 # compare sidecar TR with NIfTI header TR, noting site files where pixdim[4] uses milliseconds/units normalization may apply.
 for n,tr in zip(bs,trs):
  if isinstance(n['TR_seconds'],(int,float)) and n['TR_seconds']>0 and abs(n['TR_seconds']-tr)>0.01:
   issues.append(f'NIfTI pixdim TR {n["TR_seconds"]} differs from JSON RepetitionTime {tr}; verify NIfTI temporal units')
 status='PASS' if len(bs)==len(boldkeys) and len(ats)==len(anatkeys) and bjsonvalid and ajsonvalid and not issues else 'FAIL'
 results.append({'subject_id':sid,'site':site,'diagnosis':subj['diagnosis'],'bold_runs':len(boldkeys),'bold_shapes':'|'.join('x'.join(map(str,x['shape'])) for x in bs),'bold_voxel_size_mm':'|'.join('x'.join(map(str,x['voxel_size_mm'])) for x in bs),'timepoints':'|'.join(str(x['shape'][3]) if len(x['shape'])>=4 else '' for x in bs),'nifti_header_tr_seconds':'|'.join(str(x['TR_seconds']) for x in bs),'json_TR_seconds':'|'.join(map(str,trs)),'anatomical_shapes':'|'.join('x'.join(map(str,x['shape'])) for x in ats),'anatomical_voxel_size_mm':'|'.join('x'.join(map(str,x['voxel_size_mm'])) for x in ats),'all_nifti_headers_readable':len(bs)==len(boldkeys) and len(ats)==len(anatkeys),'gzip_integrity':'PASS' if len(bs)==len(boldkeys) and len(ats)==len(anatkeys) else 'FAIL','bold_json_valid_and_task_TR':bjsonvalid,'anatomical_json_valid':ajsonvalid,'status':status,'issues':' | '.join(issues),'scientific_image_qc':'NOT_PERFORMED_TECHNICAL_VALIDATION_ONLY'})
with (P/'qc/raw_validation.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
print('subjects',len(results),'pass',sum(x['status']=='PASS' for x in results),'fail',sum(x['status']=='FAIL' for x in results))
for r in results:
 print(r['site'],r['subject_id'],r['status'],'BOLD',r['bold_shapes'],'TRjson',r['json_TR_seconds'],'TRhdr',r['nifti_header_tr_seconds'],'anat',r['anatomical_shapes'],'issues',r['issues'])
