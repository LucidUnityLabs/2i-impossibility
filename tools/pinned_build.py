"""Resolve a user-selected TeX image once; require its digest for offline builds.

The selected image must provide python3, latexmk, pdflatex and bibtex. Registry/image
trust is a release-maintainer decision; this tool does not claim arbitrary images
are safe. Pinning happens only with an explicit --pin argument.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'build-image.lock.json'
DIGEST = re.compile(r'^[A-Za-z0-9._:/-]+@sha256:[0-9a-f]{64}$')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pin',help='explicit image tag/digest to pull, review and lock')
    p.add_argument('--platform',default='linux/amd64',help='release image platform')
    p.add_argument('--epoch',type=int)
    a = p.parse_args(argv)
    if a.pin:
        if a.pin.startswith('-') or LOCK.exists():
            raise ValueError('invalid image or existing lock; review any lock rotation explicitly')
        subprocess.run(['docker','pull','--platform',a.platform,a.pin],check=True)
        info = json.loads(subprocess.check_output(['docker','image','inspect',a.pin],text=True))[0]
        digests = info.get('RepoDigests',[])
        if len(digests) != 1 or not DIGEST.fullmatch(digests[0]):
            raise ValueError('expected one unambiguous repository digest')
        LOCK.write_text(json.dumps({'image':digests[0],'platform':info['Os']+'/'+info['Architecture']},
                                  indent=2,sort_keys=True)+'\n',encoding='utf-8')
        print(f'Pinned {digests[0]}; review image provenance before using it.')
        return
    lock = json.loads(LOCK.read_text(encoding='utf-8'))
    if not DIGEST.fullmatch(lock['image']):
        raise ValueError('the build image must be addressed by a full SHA256 digest')
    if a.epoch is None or a.epoch < 0:
        raise ValueError('--epoch is required for the pinned build')
    output = ROOT/'build/paper'
    output.mkdir(parents=True,exist_ok=True)
    if not hasattr(os,'getuid'):
        raise ValueError('this pinned runner expects a Unix Docker host')
    cmd = ['docker','run','--rm','--pull=never','--network=none','--read-only',
           '--cap-drop=ALL','--security-opt=no-new-privileges',
           '--platform',lock['platform'],'--user',f'{os.getuid()}:{os.getgid()}',
           '--env','HOME=/tmp','--env','PYTHONDONTWRITEBYTECODE=1',
           '--tmpfs','/tmp:rw,nosuid,nodev,size=1g',
           '--mount',f'type=bind,src={ROOT},dst=/src,readonly',
           '--mount',f'type=bind,src={output},dst=/src/build/paper',
           '--workdir','/src','--entrypoint','python3',lock['image'],
           'tools/build_paper.py','--epoch',str(a.epoch)]
    subprocess.run(cmd,check=True,timeout=420)


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        raise SystemExit(1)
