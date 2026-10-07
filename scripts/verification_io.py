"""Strict JSON, explicit failure, deterministic provenance, atomic output."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from exact_2i import VerificationError, require


def encode(data) -> str:
    return json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+'\n'


def atomic_write(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                dir=path.parent, prefix='.'+path.name+'.', delete=False) as f:
            name = f.name
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name,path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def provenance(script: str) -> dict:
    directory = Path(script).resolve().parent
    return {'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(directory.glob('*.py'))},
            'dependency_files_sha256':{name:hashlib.sha256((directory.parent/name).read_bytes()).hexdigest()
                for name in ('.python-version','requirements.txt','requirements.lock')
                if (directory.parent/name).is_file()},
            'arithmetic':'exact; integer, rational and algebraic operations only'}


def cli(build, script: str, argv=None) -> int:
    parser = argparse.ArgumentParser(description=build.__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--output',type=Path,help='explicit output path; atomically replaced')
    mode.add_argument('--check',type=Path,help='compare to an existing certificate; never overwrite')
    args = parser.parse_args(argv)
    try:
        report = {'schema_version':1, 'verification_status':'passed',
                  'scope':build.__doc__, 'data':build(), 'provenance':provenance(script)}
        text = encode(report)
        if args.check is not None:
            # Exact canonical bytes, not approximate floating-point comparisons.
            require(args.check.read_bytes() == text.encode('utf-8'),
                    f'certificate mismatch: {args.check}')
            print(f'PASS: {args.check}')
        elif args.output is not None:
            atomic_write(args.output,text)
            print(f'PASS: wrote {args.output}')
        else:
            print(text,end='')
        return 0
    except (VerificationError, OSError, TypeError, ValueError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        return 1
