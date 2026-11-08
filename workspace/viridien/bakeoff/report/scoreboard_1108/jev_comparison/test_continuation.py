"""Mocked continuation checks. No SSH, API, credential-file or inference IO."""
from datetime import datetime, timedelta, timezone
import importlib.util
from pathlib import Path
from unittest.mock import patch
import unittest

SPEC = importlib.util.spec_from_file_location('jev_continuation', Path(__file__).with_name('continue_jev.py'))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class FakeOps:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail
    def case(self, case):
        self.calls.append(case['case_id'])
        if case['case_id'] == self.fail:
            raise TimeoutError('uncertain transport')
    def compare_once(self):
        self.calls.append('compare')


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(case_id=f'{i:02}', suite='ext_dev') for i in range(68)]
        self.now = datetime(2026, 11, 8, 9, 0, tzinfo=timezone.utc)
        self.end = self.now + timedelta(hours=1)
        self.states = []
    def run_fake(self, ops, done, limit=68, now=None):
        return module.run_bounded(self.cases, done, ops, max_cases=limit,
                                  deadline=self.end, now=now or (lambda: self.now),
                                  checkpoint=lambda state: self.states.append(dict(state)))
    def test_skips_saved_and_compares_only_once_at_end(self):
        ops = FakeOps()
        state = self.run_fake(ops, [f'{i:02}' for i in range(65)])
        self.assertEqual(ops.calls, ['65', '66', '67', 'compare'])
        self.assertEqual(state['remaining_after_pass'], 0)
        self.assertEqual(state['status'], 'stopped')
    def test_uncertain_case_stops_without_retry_or_replay(self):
        ops = FakeOps(fail='66')
        state = self.run_fake(ops, [f'{i:02}' for i in range(65)])
        self.assertEqual(ops.calls, ['65', '66'])
        self.assertEqual(state['status'], 'inspection_required')
        self.assertEqual(state['uncertain_case'], '66')
        self.assertEqual(state['completed_this_pass'], ['65'])
    def test_deadline_stops_before_any_transport(self):
        ops = FakeOps()
        state = self.run_fake(ops, [], now=lambda: self.end)
        self.assertEqual(ops.calls, [])
        self.assertIn('cutoff', state['reason'])
    def test_rejects_unreleased_and_missing_environment_credential(self):
        with self.assertRaisesRegex(ValueError, 'single-owner release'):
            module.verify_handoff({'state': 'held'})
        receipt = {'state': 'released', 'authorized_continuation': True, 'results_sha256': 'snapshot',
                   'source_manifest_sha256': 'source', 'confirmed_by': 'manager',
                   'remote_stage_verified_idle': True, 'ready_for_api': True,
                   'prior_owner_pids': [{'pid': 12, 'creation_epoch': 0}]}
        with patch.object(module, 'sha', return_value='snapshot'), \
             patch.object(module, 'read', return_value={'source_manifest_sha256': 'source'}), \
             patch.object(module.psutil, 'pid_exists', return_value=False), \
             patch.object(module.psutil, 'process_iter', return_value=[]), \
             patch.dict(module.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'Environment-only'):
                module.verify_handoff(receipt)

if __name__ == '__main__':
    unittest.main(verbosity=2)
