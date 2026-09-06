# 2i-impossibility

Supporting code for the preprint "Structural Obstructions to
Standard-Model Derivation from Binary Icosahedral Orbifolds and
Modular-Flavor Extensions" (see `paper/paper1.pdf`).

The paper exhaustively tests the class of candidate Standard Model
derivations based on E8 gauge theory compactified on
M4 x S3/2I x T2(tau=i), where 2I = SL(2,F5) is the binary icosahedral
group, together with natural extensions (modular-flavor symmetries,
eclectic non-split groups, twisted modular tensor categories, 2-group
constructions). The result is negative: three impossibility theorems,
each closing the program from an independent mathematical direction,
plus five supporting propositions.

## The three theorems

**Theorem A (Arithmetic obstruction).** Characters of finite-dimensional
representations of any finite group G lie in the cyclotomic field
Q(zeta_exp(G)). For 2I the exponent is 60 with prime support {2,3,5},
while sin(pi/14) lies in Q(zeta_14) and requires the prime 7. By
Goursat's lemma and the Bantay--Coste--Gannon--Ruelle character cap this
prime separation persists through all known finite-symmetry constructions,
so no 2I-based character-theoretic construction can reproduce the
Cabibbo-angle value sin(pi/14).

**Theorem B (Frobenius--Schur chirality obstruction).** Every faithful
irreducible representation of 2I is quaternionic (Frobenius--Schur
indicator -1) and no irreducible is of complex type. Consequently the
equivariant Dirac index of any 2I worldsheet orbifold vanishes
identically, independent of level matching, gauge embedding, and modular
invariance, ruling out chiral spectra from this mechanism.

**Theorem C (F4 x G2 centralizer obstruction).** F4 x G2 is a maximal
regular dual pair in E8 with trivial centralizer, so no heterotic bundle
construction can realize it as the visible 4D gauge group; the natural
2I Wilson-line embeddings land in the wrong centralizer structure to
produce the Standard Model gauge group.

## Status

Preprint. Not yet peer reviewed.

## Verification

Every computational claim is reproduced from scripts with no external
data files. The main verifier (numpy only) checks 20 assertions covering
the 2I construction, conjugacy classes, the character table,
Frobenius--Schur indicators (Theorem B), and the arithmetic obstruction
(Theorem A):

    python3 scripts/supplement_verify.py

Expected final line: `SUMMARY: ALL 20 CHECKS PASS`. A saved run is in
`scripts/supplement_verify_output.txt`.

Per-theorem evidence scripts (each self-contained; some additionally use
sympy/mpmath; each writes a `_results.json` next to itself when run):

- `scripts/verify_2I_reps.py` -- exact (sympy) representation theory of
  2I: character table, Frobenius--Schur indicators, faithful irreps.
- `scripts/gamma70_fusion.py` -- Gamma(70) modular-flavor fusion rules
  and the modular-tower plateau.
- `scripts/eclectic_nonsplit.py` -- eclectic non-split extension search
  (Goursat-type constructions attempting to evade Theorem A).
- `scripts/compactification_search.py` -- 600-cell chirality /
  compactification paradigm search (Theorem B stress test).
- `scripts/wilson_line_centralizer.py` -- Wilson-line centralizer
  computation in E8 (Theorem C).

## Paper

`paper/paper1.pdf` (source: `paper/paper1.tex`, bibliography:
`paper/references.bib`).

## Citation

No DOI yet. Cite this repository:

    Tyler, "Structural Obstructions to Standard-Model Derivation from
    Binary Icosahedral Orbifolds and Modular-Flavor Extensions,"
    https://github.com/im-tyler/2i-impossibility (2026).

## License

MIT -- see `LICENSE`.
