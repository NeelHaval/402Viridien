import copy
import tempfile
import unittest
from pathlib import Path

import registry_check_private as check


class RegistryCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = check.load_pinned(check.REGISTRY, check.REGISTRY_SHA)
        cls.binding = check.load_pinned(check.BINDING, check.BINDING_SHA)
        cls.metadata = check.load_pinned(Path(cls.binding['sourceMetadataPath']), cls.binding['sourceMetadataSha256'])

    def test_valid_snapshot_is_not_launch_authority(self):
        result = check.validate(self.registry, self.binding, self.metadata)
        self.assertEqual(result['counts'], check.COUNTS)
        self.assertFalse(result['api_launch_authorized'])

    def test_identity_and_launch_flag_tampering_fail(self):
        for field, value in [('imageSha256', 'bad'), ('window', [0, 1]), ('apiLaunchAllowed', True)]:
            altered = copy.deepcopy(self.binding)
            altered['entries'][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                check.validate(self.registry, altered, self.metadata)

    def test_protected_ancestor_and_missing_case_fail(self):
        altered = copy.deepcopy(self.binding)
        allowed = next(item for item in altered['entries'] if item['decision'] == 'allow')
        allowed['ancestorGroupIds'].append(self.registry['protectedGroups'][0])
        with self.assertRaises(ValueError):
            check.validate(self.registry, altered, self.metadata)
        altered = copy.deepcopy(self.binding)
        altered['entries'].pop()
        with self.assertRaises(ValueError):
            check.validate(self.registry, altered, self.metadata)

    def test_pin_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.json'
            path.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                check.load_pinned(path, check.BINDING_SHA)


if __name__ == '__main__':
    unittest.main(verbosity=2)
