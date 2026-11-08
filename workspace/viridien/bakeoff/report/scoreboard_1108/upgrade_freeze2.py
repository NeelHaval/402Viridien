from pathlib import Path
p=Path(__file__).with_name('score_all.py');s=p.read_text(encoding='utf-8')
s=s.replace("B/'strategies'/'s02_features_pelt.py']", "B/'strategies'/'s02_features_pelt.py',B/'emil'/'paths.py']")
s=s.replace('def validate_records(records,manifest):','def validate_records(records,manifest,allow_legacy=False):')
s=s.replace("if record.get('manifest_digest') not in (None,manifest_digest(manifest)):", "if record.get('manifest_digest')!=manifest_digest(manifest) and not (allow_legacy and record.get('manifest_digest') is None):")
pos=s.index('\ndef build_manifest')
s=s[:pos]+'''
def freeze_metadata():
    packages={}
    for name in ('torch','transformers','numpy','scipy','scikit-learn','Pillow','opencv-python','rapidocr-onnxruntime','easyocr','ruptures','tifffile'):
        try:packages[name]=package_version(name)
        except Exception:packages[name]='not installed under this distribution name'
    return dict(python=platform.python_version(),executable=sys.executable,packages=packages,
        dino=dict(model='facebook/dinov2-small',weights_sha256='ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1',config_sha256='1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d'),
        note='Pinned from pre-run CUDA smoke. Runtime DINO diagnostic hashes must match. First six legacy records were input-validated at migration, not hash-checked at their original read time.')
''' + s[pos:]
s=s.replace("manifest=dict(version=1,cases=cases,", "manifest=dict(version=2,freeze=freeze_metadata(),cases=cases,")
s=s.replace("'preprocess_seconds','end_to_end_seconds'", "'preprocess_seconds','end_to_end_seconds'")
p.write_text(s,encoding='utf-8')
