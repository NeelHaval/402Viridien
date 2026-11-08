"""Protect split boundaries and release installation paths."""
import copy
import hashlib
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import workspace


class DataBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = workspace.read(workspace.DATA/'real/manifest.json')

    def test_published_real_split_is_consistent(self):
        self.assertEqual(workspace.audit_real(self.manifest), [])

    def test_exact_test_pixels_cannot_enter_training(self):
        data = copy.deepcopy(self.manifest)
        train = next(r for r in data['records'] if r['usable_for_generation'])
        test = next(r for r in data['records'] if r['split'] == 'test')
        train['pixel_sha256'] = test['pixel_sha256']
        self.assertTrue(any('test pixels' in e for e in workspace.audit_real(data)))

    def test_quality_hold_cannot_enter_generation(self):
        data = copy.deepcopy(self.manifest)
        row = next(r for r in data['records'] if r['quality'] == 'review')
        row['usable_for_generation'] = True
        self.assertTrue(any('generator permission' in e for e in workspace.audit_real(data)))

    def test_well_cannot_cross_splits(self):
        data = copy.deepcopy(self.manifest)
        train = next(r for r in data['records'] if r['split'] == 'train')
        test = next(r for r in data['records'] if r['split'] == 'test')
        train['source_group'] = test['source_group']
        self.assertTrue(any('both splits' in e for e in workspace.audit_real(data)))

    def test_public_suites_have_complete_disjoint_inputs(self):
        root = workspace.ROOT/'viridien'
        registry = workspace.read(root/'bakeoff/harness/suites.json')['suites']
        log_sets = []
        for suite in ('ext_shared', 'ext_shared_verify'):
            config = registry[suite]
            dataset = root/config['dataset']['default']
            truth = root/config['truth']['default']
            items = workspace.read(root/config['items']['default'])
            self.assertTrue(items)
            logs = {item['log'] for item in items}
            log_sets.append(logs)
            for log in logs:
                self.assertTrue((dataset/'images'/(log+'.tif')).is_file(), log)
                self.assertTrue((dataset/'bbox_annotation'/(log+'.json')).is_file(), log)
                self.assertTrue((truth/(log+'.csv')).is_file(), log)
        self.assertFalse(log_sets[0] & log_sets[1])


class ArchiveBoundaryTests(unittest.TestCase):
    def check_archive(self, names, root, bundle='real'):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w') as archive:
            for name in names:
                archive.writestr(name, b'sample')
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            return workspace.archive_members(archive, bundle, root)

    def test_traversal_absolute_drive_and_wrong_bundle_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in ['../outside', '/outside', 'C:/outside',
                         'viridien/data/real/../../outside',
                         'viridien/data/real/C:/outside',
                         'viridien/pattern_lab/data/live.sqlite3']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    self.check_archive([name], Path(folder))

    def test_different_local_work_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'viridien/data/real/manifest.json'
            path.parent.mkdir(parents=True)
            path.write_text('local work', encoding='utf-8')
            with self.assertRaises(ValueError):
                self.check_archive(['viridien/data/real/manifest.json'], Path(folder))
            self.assertEqual(path.read_text(encoding='utf-8'), 'local work')

    def test_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                info = zipfile.ZipInfo('viridien/data/real/link')
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, '/outside')
            buffer.seek(0)
            with zipfile.ZipFile(buffer) as archive, self.assertRaises(ValueError):
                workspace.archive_members(archive, 'real', Path(folder))

    def test_valid_archive_stays_inside_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = self.check_archive(['viridien/data/real/objects/a.png'], Path(folder))
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0][1].is_relative_to(Path(folder).resolve()))

    def test_download_uses_the_catalog_repository_and_checks_the_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                archive.writestr('viridien/data/real/objects/example.txt', b'checked input')
            payload = buffer.getvalue()
            asset = {'file': 'real-labels.zip', 'bytes': len(payload),
                     'sha256': hashlib.sha256(payload).hexdigest(), 'status': 'test fixture'}
            catalog = {'repository': 'NeelHaval/viridian402',
                       'tag': 'workspace-20261108-v1', 'bundles': {'real': asset}}

            def fetch(args, **kwargs):
                self.assertEqual(args[args.index('--repo') + 1], catalog['repository'])
                self.assertEqual(args[3], catalog['tag'])
                (Path(args[args.index('--dir') + 1])/asset['file']).write_bytes(payload)

            with patch.object(workspace, 'ROOT', root), \
                    patch.object(workspace, 'read', return_value=catalog), \
                    patch.object(workspace.shutil, 'which', return_value='gh'), \
                    patch.object(workspace.subprocess, 'run', side_effect=fetch) as run:
                self.assertEqual(workspace.download('real'), 0)
            run.assert_called_once()
            self.assertEqual((root/'viridien/data/real/objects/example.txt').read_bytes(), b'checked input')


if __name__ == '__main__':
    unittest.main()
