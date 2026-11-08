from pathlib import Path
p=Path(__file__).with_name('score_all.py');s=p.read_text(encoding='utf-8').replace("rec['strategy_diagnostics'].get('weights_sha256')", "rec['strategy_diagnostics'].get('model',{}).get('weights_sha256')").replace("rec['strategy_diagnostics'][key]!=manifest['freeze']['dino'][key]", "rec['strategy_diagnostics']['model'][key]!=manifest['freeze']['dino'][key]")
p.write_text(s,encoding='utf-8')
