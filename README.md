# 2i-impossibility: scoped exact verification

This repository contains exact finite checks of binary-icosahedral representation
and invariant theory, selected SU(2) maps into E6/E8, congruence-group arithmetic,
and explicit counterexamples to several overly broad exclusions in the original
preprint. These computations do **not** establish a universal impossibility theorem
for Standard-Model constructions.

The original `paper/paper1.pdf` must not be treated as validated by the corrected
scripts until its claims have been revised and a reviewed replacement PDF has
been explicitly published. `paper/paper1.revised.tex` is a proposed, narrower
replacement source; it requires author review before promotion.

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

The wheel downloads were not performed during this audit; the hashes were checked
against PyPI, and execution used the installed versions stated above. Hash locks
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

The result is `build/paper/paper1.pdf`, not the tracked published PDF. The adjacent
build report records source hashes, tool versions, source date and PDF hash.

For a release, select and review an OCI image containing those tools, then resolve
its immutable digest explicitly with `python tools/pinned_build.py --pin IMAGE`.
Commit `build-image.lock.json`, archive the image by digest, and run:

```sh
make paper-pinned
```

The pinned runner has networking disabled, drops capabilities, uses an unprivileged
user and mounts sources read-only. An image digest is necessary but is not a
publisher trust audit. Local `make paper` alone does not pin TeX, fonts or the OS.
The GitHub Python workflow is a correctness gate, not a claim of bit-reproducible
hosted-runner environments. The proposed TeX workflow must be exercised after
choosing the actual image; no container digest is fabricated in this repository.

Before publishing, inspect the PDF, review the mathematical statements and then
copy the reviewed build explicitly to `paper/paper1.pdf`. Preserve prior published
versions through Git/release history rather than rewriting history.

## License and citation

MIT, as in `LICENSE`. Cite the precise repository commit and the actual version of
the paper used. No DOI or peer-review status is implied by successful scripts.
