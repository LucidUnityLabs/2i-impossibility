"""Run every scoped certificate producer; updates require explicit --write."""
import argparse
import importlib
from pathlib import Path
from verification_io import cli

MODULES = ('supplement_verify','verify_2I_reps','wilson_line_centralizer',
           'gamma70_fusion','eclectic_nonsplit','compactification_search')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write',action='store_true')
    mode.add_argument('--check',action='store_true')
    p.add_argument('--directory',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    a = p.parse_args(argv)
    failures = 0
    for name in MODULES:
        module = importlib.import_module(name)
        failures += cli(module.build,module.__file__,
                        ['--output' if a.write else '--check',str(a.directory/(name+'.json'))]) != 0
    return int(failures != 0)


if __name__ == '__main__':
    raise SystemExit(main())
