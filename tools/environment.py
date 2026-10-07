"""Record the execution environment separately from deterministic mathematical data."""
import importlib.metadata
import json
import platform
import sys


def main():
    result = {'python_version':platform.python_version(),
              'python_build':sys.version, 'implementation':platform.python_implementation(),
              'platform':platform.platform(),
              'packages':{name:importlib.metadata.version(name) for name in ('sympy','mpmath')}}
    print(json.dumps(result,sort_keys=True,indent=2))


if __name__ == '__main__':
    main()
