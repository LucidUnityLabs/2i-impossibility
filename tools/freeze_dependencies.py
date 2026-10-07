"""Explicit online bootstrap: download pinned wheels, verify metadata, write a hash lock.

The downloaded wheelhouse must be archived alongside requirements.lock for offline
reproduction. This is a trust-on-first-download bootstrap, not an independent
validation of package publishers. Never use it as part of a certificate check.
"""
from __future__ import annotations
import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def normalized(name):
    return re.sub(r'[-_.]+','-',name).lower()


def wheel_metadata(path):
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
        if len(names) != 1:
            raise ValueError(f'bad metadata count in {path.name}')
        msg = BytesParser().parsebytes(z.read(names[0]))
    return normalized(msg['Name']),msg['Version']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheelhouse',type=Path,default=ROOT/'wheelhouse')
    p.add_argument('--lock',type=Path,default=ROOT/'requirements.lock')
    a = p.parse_args(argv)
    # Do not quietly rotate an existing trust baseline.
    if a.wheelhouse.exists() or a.lock.exists():
        raise ValueError('refusing to overwrite an existing wheelhouse or lock; review/remove explicitly')
    pins = {}
    for line in (ROOT/'requirements.txt').read_text(encoding='utf-8').splitlines():
        line = line.split('#',1)[0].strip()
        if line:
            if not re.fullmatch(r'[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.+-]*',line):
                raise ValueError(f'only exact pins allowed: {line}')
            name,version = line.split('==')
            pins[normalized(name)] = version
    with tempfile.TemporaryDirectory(prefix='2i-wheels-') as temp:
        subprocess.run([sys.executable,'-m','pip','--isolated','download',
            '--index-url','https://pypi.org/simple','--only-binary=:all:','--no-deps',
            '--dest',temp,*[f'{k}=={v}' for k,v in sorted(pins.items())]],check=True)
        wheels = sorted(Path(temp).glob('*.whl'))
        entries = {}
        for wheel in wheels:
            name,version = wheel_metadata(wheel)
            if pins.get(name) != version or name in entries:
                raise ValueError('downloaded wheel metadata does not match the requested pins')
            # Both current dependencies are portable Python wheels. Fail on an unexpected build.
            if not wheel.name.endswith('-none-any.whl'):
                raise ValueError('expected platform-independent wheels for this baseline')
            entries[name] = (wheel,hashlib.sha256(wheel.read_bytes()).hexdigest())
        if set(entries) != set(pins):
            raise ValueError('missing or extra wheels')
        a.wheelhouse.mkdir(parents=True)
        a.lock.parent.mkdir(parents=True,exist_ok=True)
        lines = ['# Generated from reviewed, archived wheels; do not hand-edit hashes.']
        for name,(wheel,digest) in sorted(entries.items()):
            (a.wheelhouse/wheel.name).write_bytes(wheel.read_bytes())
            lines.append(f'{name}=={pins[name]} --hash=sha256:{digest}')
        a.lock.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Wrote {a.lock}; archive {a.wheelhouse} separately and review both wheels.')


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,subprocess.CalledProcessError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        raise SystemExit(1)
