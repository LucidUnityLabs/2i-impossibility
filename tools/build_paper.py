"""Build twice from clean copies; fail on errors, unresolved references or PDF byte drift.

This controls source-date metadata but does NOT pin a TeX distribution. Use the
pinned OCI runner for a hermetic release environment. No published PDF is replaced.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('latexmk','pdflatex','bibtex')
BAD_LOG = re.compile(r'(?:undefined (?:references|citations)|(?:Reference|Citation) .+ undefined|'
                     r'Label\(s\) may have changed|multiply[- ]defined labels|'
                     r'I (?:couldn.t open database file|didn.t find a database entry)|'
                     r'^!|Emergency stop|Fatal error)',re.I|re.M)
WRAPPER = r'''\ifdefined\pdfinfoomitdate\pdfinfoomitdate=1\fi
\ifdefined\pdftrailerid\pdftrailerid{}\fi
\ifdefined\pdfsuppressptexinfo\pdfsuppressptexinfo=15\fi
\input{paper1.tex}
'''


def run_one(source,work,env):
    shutil.copytree(source,work,ignore=shutil.ignore_patterns(
        'paper1.pdf','*.aux','*.bbl','*.blg','*.log','*.out','*.fls','*.fdb_latexmk',
        '*.synctex.gz','*.toc','*.lof','*.lot'))
    wrapper = work/'repro-entry.tex'
    wrapper.write_text(WRAPPER,encoding='utf-8')
    command = ['latexmk','-norc','-pdf','-interaction=nonstopmode',
               '-halt-on-error','-file-line-error','-recorder','-jobname=paper1',
               '-pdflatex=pdflatex %O -no-shell-escape %S',wrapper.name]
    subprocess.run(command,cwd=work,env=env,check=True,timeout=180)
    for suffix in ('log','blg'):
        path = work/('paper1.'+suffix)
        if not path.exists():
            raise ValueError(f'missing {path.name}: bibliography/build was not completed')
        text = path.read_text(encoding='utf-8',errors='replace')
        if BAD_LOG.search(text):
            raise ValueError(f'unresolved or fatal diagnostic in {path.name}')
    pdf = work/'paper1.pdf'
    if not pdf.exists():
        raise ValueError('missing PDF output')
    return pdf.read_bytes()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch',type=int,default=None)
    p.add_argument('--output',type=Path,default=ROOT/'build/paper/paper1.pdf')
    a = p.parse_args(argv)
    if a.output.resolve() == (ROOT/'paper/paper1.pdf').resolve():
        raise ValueError('refusing to overwrite the published PDF; review/promote explicitly')
    for tool in REQUIRED:
        if shutil.which(tool) is None:
            raise ValueError(f'required build tool not found: {tool}')
    if not (ROOT/'paper/paper1.tex').is_file() or not (ROOT/'paper/references.bib').is_file():
        raise ValueError('paper/paper1.tex and paper/references.bib are required')
    epoch = a.epoch
    if epoch is None:
        epoch = int(subprocess.check_output(['git','log','-1','--format=%ct'],cwd=ROOT,text=True).strip())
    if epoch < 0:
        raise ValueError('SOURCE_DATE_EPOCH must be nonnegative')
    env = dict(os.environ,SOURCE_DATE_EPOCH=str(epoch),FORCE_SOURCE_DATE='1',TZ='UTC',LC_ALL='C')
    versions = {t:next(line for line in subprocess.check_output([t,'--version'],text=True).splitlines()
                         if line.strip()) for t in REQUIRED}
    with tempfile.TemporaryDirectory(prefix='2i-tex-') as temp:
        base = Path(temp)
        pdf_a = run_one(ROOT/'paper',base/'first',env)
        pdf_b = run_one(ROOT/'paper',base/'second',env)
        if pdf_a != pdf_b:
            raise ValueError('two clean builds produced different PDF bytes; no output promoted')
        a.output.parent.mkdir(parents=True,exist_ok=True)
        temp_output = a.output.with_suffix('.pdf.tmp')
        temp_output.write_bytes(pdf_a)
        temp_output.replace(a.output)
        report = {'source_date_epoch':epoch,'tools':versions,'two_clean_builds_equal':True,
                  'pdf_sha256':hashlib.sha256(pdf_a).hexdigest(),
                  'source_sha256':{str(f.relative_to(ROOT/'paper')):hashlib.sha256(f.read_bytes()).hexdigest()
                       for f in sorted((ROOT/'paper').rglob('*')) if f.is_file() and f.suffix in ('.tex','.bib')}}
        a.output.with_suffix('.build.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(f'PASS: reproducible clean builds; wrote {a.output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        print(f'FAIL: {exc}',file=sys.stderr)
        raise SystemExit(1)
