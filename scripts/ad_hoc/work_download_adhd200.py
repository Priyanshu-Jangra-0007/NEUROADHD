import concurrent.futures
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

BUCKET = 'https://fcp-indi.s3.amazonaws.com/'
PREFIX = 'data/Projects/ADHD200/RawDataBIDS/'
OUT = Path('/Users/apple/Documents/Codex/2026-09-29/down/outputs/ADHD200_RawDataBIDS')
NS = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}


def list_all():
    objects = []
    token = None
    while True:
        params = {'list-type': '2', 'prefix': PREFIX}
        if token:
            params['continuation-token'] = token
        url = BUCKET + '?' + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.findall('s3:Contents', NS):
            key = item.findtext('s3:Key', namespaces=NS)
            size = int(item.findtext('s3:Size', namespaces=NS) or 0)
            if key and not key.endswith('/'):
                objects.append((key, size))
        truncated = root.findtext('s3:IsTruncated', namespaces=NS) == 'true'
        if not truncated:
            break
        token = root.findtext('s3:NextContinuationToken', namespaces=NS)
        if not token:
            raise RuntimeError('S3 listing was truncated but no continuation token was returned')
    return objects


def fetch(entry):
    key, size = entry
    rel = key[len(PREFIX):]
    dest = OUT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return size, 0, 'cached'
    url = BUCKET + urllib.parse.quote(key, safe='/')
    last_error = None
    for attempt in range(5):
        tmp = dest.with_name(dest.name + '.part')
        try:
            with urllib.request.urlopen(url, timeout=180) as response, open(tmp, 'wb') as target:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    target.write(block)
            if tmp.stat().st_size != size:
                raise IOError(f'size mismatch for {key}: expected {size}, got {tmp.stat().st_size}')
            os.replace(tmp, dest)
            return size, size, 'downloaded'
        except Exception as error:
            last_error = error
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
            time.sleep(min(2 ** attempt, 30))
    return size, 0, f'ERROR {key}: {last_error}'


OUT.mkdir(parents=True, exist_ok=True)
print('Listing complete RawDataBIDS object set...', flush=True)
objects = list_all()
total = sum(size for _, size in objects)
print(f'Found {len(objects)} files totaling {total / (1024**3):.2f} GiB', flush=True)
manifest = OUT.parent / 'ADHD200_RawDataBIDS_manifest.tsv'
with manifest.open('w') as f:
    f.write('key\tsize_bytes\n')
    for key, size in objects:
        f.write(f'{key}\t{size}\n')
completed = 0
errors = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    for i, result in enumerate(pool.map(fetch, objects), 1):
        expected, got, status = result
        completed += got
        if status.startswith('ERROR'):
            errors.append(status)
        if i % 100 == 0 or i == len(objects):
            print(f'Progress: {i}/{len(objects)} files; {completed / (1024**3):.2f} GiB fetched; errors={len(errors)}', flush=True)
if errors:
    error_path = OUT.parent / 'ADHD200_RawDataBIDS_download_errors.txt'
    error_path.write_text('\n'.join(errors) + '\n')
    print(f'Finished with {len(errors)} errors; details: {error_path}', flush=True)
else:
    print('Download complete: all listed files present and size-verified.', flush=True)
print(f'Dataset directory: {OUT}', flush=True)
print(f'Manifest: {manifest}', flush=True)
