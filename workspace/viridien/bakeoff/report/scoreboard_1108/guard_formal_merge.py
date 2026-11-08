from pathlib import Path
p=Path(__file__).with_name('merge_final_shards.py');s=p.read_text(encoding='utf-8').replace("HERE/'bounded_controller_receipt.json')", "HERE/'bounded_controller_receipt.json',HERE/'formal_controller_receipt.json')");p.write_text(s,encoding='utf-8')
