import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'tools'/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildToolTests(unittest.TestCase):
    def test_tex_fatal_diagnostics(self):
        b = load('build_paper')
        for text in ('LaTeX Warning: There were undefined references.',
                     "LaTeX Warning: Citation `missing' on page 1 undefined",
                     'LaTeX Warning: Label(s) may have changed.',
                     'I couldn\'t open database file references.bib',
                     '! Undefined control sequence.'):
            self.assertIsNotNone(b.BAD_LOG.search(text),text)
        self.assertIsNone(b.BAD_LOG.search('Output written on paper1.pdf (12 pages).'))

    def test_missing_tex_tool_fails_early(self):
        b = load('build_paper')
        with patch.object(b.shutil,'which',return_value=None):
            with self.assertRaisesRegex(ValueError,'required build tool'):
                b.main(['--epoch','0'])

    def test_published_pdf_never_overwritten(self):
        b = load('build_paper')
        with self.assertRaisesRegex(ValueError,'published PDF'):
            b.main(['--output',str(ROOT/'paper/paper1.pdf'),'--epoch','0'])

    def test_container_digest_required(self):
        b = load('pinned_build')
        self.assertIsNone(b.DIGEST.fullmatch('example/tex:latest'))
        self.assertIsNotNone(b.DIGEST.fullmatch('example/tex@sha256:'+'a'*64))

    def test_dependency_bootstrap_refuses_overwrite(self):
        b = load('freeze_dependencies')
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'refusing to overwrite'):
                b.main(['--wheelhouse',temp,'--lock',str(Path(temp)/'requirements.lock')])
