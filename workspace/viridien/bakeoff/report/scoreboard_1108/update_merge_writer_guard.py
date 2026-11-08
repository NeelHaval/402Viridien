from pathlib import Path
p=Path(__file__).with_name('merge_final_shards.py');s=p.read_text(encoding='utf-8')
a=s.index("    receipt=read(HERE/'local_remaining_receipt.json')")
b=s.index("    manifest=m.read",a)
s=s[:a]+'''    for receipt_path in (HERE/'local_remaining_receipt.json',HERE/'bounded_controller_receipt.json'):
        if not receipt_path.exists():continue
        receipt=read(receipt_path)
        try:
            proc=psutil.Process(receipt['pid']);born=datetime.fromisoformat(receipt['creation_utc'].replace('Z','+00:00')).timestamp()
            if abs(proc.create_time()-born)<1 and proc.is_running():raise RuntimeError('Owned local writer/controller is still active; merge is blocked')
        except psutil.NoSuchProcess:pass
''' + s[b:]
p.write_text(s,encoding='utf-8')
