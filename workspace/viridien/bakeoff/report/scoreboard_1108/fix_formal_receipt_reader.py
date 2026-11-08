from pathlib import Path
p=Path(__file__).with_name('formal_shenzi_recheck.py');s=p.read_text(encoding='utf-8').replace("receipt=m.read(H/'bounded_controller_receipt.json')", "receipt=json.loads((H/'bounded_controller_receipt.json').read_text(encoding='utf-8-sig'))");p.write_text(s,encoding='utf-8')
