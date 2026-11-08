"""Wait for the restarted lab to finish its first successful build."""
import json
import argparse
import time
import urllib.error
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--timeout', type=int, default=600)
args = parser.parse_args()
deadline = time.monotonic() + args.timeout
last = None
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8781/state.json', timeout=5) as response:
            state = json.load(response)
        last = state.get('status', {})
        if last.get('error'):
            raise SystemExit('FAIL readiness: ' + last['error'])
        if last.get('builds', 0) >= 1 and not last.get('building'):
            print('PASS readiness:', json.dumps(last))
            print('patterns:', state['pattern_count'], 'human:', state['human_pattern_count'],
                  'machine-verified:', state['verified_pattern_count'],
                  'borrowed:', state.get('borrowed_pattern_count', 0),
                  'swatch:', state.get('swatch_pattern_count', 0),
                  'agent-made:', state.get('agent_made_pattern_count', 0),
                  'unlisted:', state.get('unlisted_pattern_count', 0), 'seconds:', state['seconds'])
            break
    except (OSError, urllib.error.URLError) as error:
        last = str(error)
    time.sleep(1)
else:
    raise SystemExit('FAIL readiness timeout: ' + str(last))
