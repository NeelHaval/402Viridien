# Services and ports

| Service | VM port | Public access | Start / owner |
| --- | --- | --- | --- |
| Lithology labelling | 8765 | localhost only | `cd labelling && source .venv/bin/activate && python scripts/label_server.py`; owner: emilabos |
| SSH | 22 | `213.32.26.26:19036` | system ssh; owner |
| Merge manager | none | `htb status` | `htb-manager.timer`, every 20 min; user |
| Pattern labeller | 8776 | localhost by default; LAN sharing is optional | `cd viridien/pattern_lab && python server.py --host 127.0.0.1 --port 8776`; owner: silmoon04 |
| Pattern generator | 8781 | localhost | `cd viridien/pattern_lab && python -m genlab.server`; owner: silmoon04 |
| Generator progress board | 8782 | localhost | `cd viridien/pattern_lab && python data/genlab/fleet/board/board.py`; owner: silmoon04 |
| Legend atlas | 8783 | localhost | `cd viridien/pattern_lab && python data/genlab/atlas/atlas.py`; owner: silmoon04 |
| Account sync | none | none | root cron, every 5 min |

Document new services here with their command, port and responsible teammate. For private development ports use `ssh -L 3000:127.0.0.1:3000 -p 19036 LOGIN@213.32.26.26` and browse localhost:3000. Bind development servers to localhost where practical.
