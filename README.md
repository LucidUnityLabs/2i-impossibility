# 2i-impossibility: scoped exact verification

This repository contains exact finite checks of binary-icosahedral representation
and invariant theory, selected SU(2) maps into E6/E8, congruence-group arithmetic,
and explicit counterexamples to several overly broad exclusions in the original
preprint. These computations do **not** establish a universal impossibility theorem
for Standard-Model constructions.

The corrected paper is `paper/paper1.revised.pdf`, built from
`paper/paper1.revised.tex`. Its adjacent `.build.json` records source hashes,
tool versions and two-clean-build equality. The historical `paper/paper1.tex`
and `paper/paper1.pdf` are preserved unchanged and contain superseded claims;
corrected certificates do not validate that baseline.

## Tested Python baseline

Python 3.13.5, SymPy 1.14.0 and mpmath 1.3.0. These are tested version pins, not a
claim that they are the newest releases. The binary-icosahedral core uses only
Python integers. SymPy is used for exact cyclotomic and class-algebra calculations.
There is no NumPy eigensolver or floating-point acceptance tolerance.

For an initial exploratory installation:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
make test
```

The included `requirements.lock` pins the two portable wheels by SHA256, as
published on their official PyPI release-file pages. Download and archive them:

```sh
make fetch-wheels
make install-locked
```

The wheel hashes were checked against PyPI. Use the commands above to acquire
the exact portable artifacts; tests use the installed versions stated above. Hash locks
prevent artifact drift but do not independently authenticate publishers. Preserve
the wheelhouse in a release archive for offline reproduction; it is ignored by Git.
For an intentional dependency update, edit the version pins and use
`make bootstrap-lock` to create a new lock and wheelhouse under `build/`, then
review them before replacing the committed lock. Existing trust baselines are
never silently overwritten. Update the tested Python baseline and rerun every
certificate check when changing dependencies.

## Verification and certificates

```sh
make test        # normal interpreter and python -O
make verify      # compare current results and source hashes to committed results
```

An intentional update is separate from verification:

```sh
make regenerate
# Review every results/*.json diff before committing it.
```

Each script accepts `--output FILE` to write or `--check FILE` to compare. With
neither option, it prints canonical JSON to standard output. Output paths do not
depend on the working directory; explicitly supplied relative paths have ordinary
command-line semantics. Failures return nonzero and never publish a new PASS
certificate. `--check` never overwrites its input. Results contain exact integers
or explicitly encoded algebraic values, stable labels, source/dependency-file hashes and scope
limitations. A passing finite check is not a passing verdict on an uncomputed
physical theory.

| Script | Actual verified scope |
|---|---|
| `supplement_verify.py` | Exact 2I character/FS table, doublet Molien coefficients, cubic arithmetic |
| `verify_2I_reps.py` | Faithful real-eight module enumeration with constructed spin lifts; tensor invariants |
| `wilson_line_centralizer.py` | Specified principal/regular maps and the stated j=3/2 block map; no exhaustive scan |
| `gamma70_fusion.py` | Exact CRT, SL2(7) characters/fields, Gamma70 dimensions/FS data and modular dimensions |
| `eclectic_nonsplit.py` | Exact finite examples and counterexamples; undefined categories/groups remain unresolved |
| `compactification_search.py` | Topological arithmetic; unsupported catalog entries explicitly quarantined |

Changing any verifier source changes the provenance of every certificate. This is
conservative by design. Review results and source changes together, rather than
blindly accepting regenerated files. Certificate bytes are deterministic in the
tested environment; this is not a proof-assistant formalization or a guarantee
across untested Python/SymPy releases.

## Paper build

The local build requires Python, latexmk, pdfLaTeX, BibTeX and all packages used in
the paper. It builds twice from clean copies, rejects unresolved references and
requires byte-identical PDFs:

```sh
make paper
```

The default source is `paper/paper1.revised.tex` and the result is
`build/paper/paper1.revised.pdf`. Historical builds require the explicit
`--source paper1.tex` option. The builder refuses any tracked paper-directory
output and rejects overfull boxes as well as unresolved references. The adjacent
build report records source hashes, tool versions, source date and PDF hash.

`build-image.lock.json` pins the Linux amd64 image from
[xu-cheng/latex-docker](https://github.com/xu-cheng/latex-docker) by its
platform-specific OCI manifest digest, verified against downloaded registry
manifest bytes. The image's package set supplies Python and the required TeX
tools. Run:

```sh
docker pull --platform linux/amd64 "$(python3 -c 'import json; print(json.load(open("build-image.lock.json"))["image"])')"
make paper-pinned
```

The pinned runner has networking disabled, drops capabilities, uses an unprivileged
user and mounts sources read-only. The GitHub workflow includes this two-build
paper gate. Local macOS/TeX Live 2026 builds passed; pinned-container execution
and image archival remain release requirements. The prepared pin and workflow
are not a claim of a successful container build. Archive the image by digest
with the eventual release, and require passing hosted verification and paper
jobs for the published revision.

Reviewed corrected PDFs are promoted explicitly to `paper/paper1.revised.pdf`;
the historical filenames remain intact. No DOI or new peer-review status follows
from the technical review.

## License and citation

MIT, as in `LICENSE`. Cite the precise repository commit and the actual version of
the paper used. No DOI or peer-review status is implied by successful scripts.
