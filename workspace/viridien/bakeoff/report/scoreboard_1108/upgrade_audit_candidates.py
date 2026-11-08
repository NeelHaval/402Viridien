from pathlib import Path
p=Path(__file__).with_name('audit_run_integrity.py');s=p.read_text(encoding='utf-8').replace("candidates=['s19_combine','s20_v4','s21_v4dino']", "candidates=manifest['candidates']");p.write_text(s,encoding='utf-8')
