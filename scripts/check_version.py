"""Reject a version tag that would publish a differently named package."""
import argparse
from pathlib import Path
import tomllib


def check_version(tag,project_file):
    version=tomllib.loads(Path(project_file).read_text(encoding='utf-8-sig'))['project']['version']
    if tag != 'v'+version:
        raise ValueError(f'Release tag {tag!r} does not match package version v{version}')
    return version


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--project-file',default='pyproject.toml')
    args=parser.parse_args()
    print(check_version(args.tag,args.project_file))
