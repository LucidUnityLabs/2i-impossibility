# Historical baseline outputs (pre-audit revision 49aaee56)

These five files are the committed outputs of the ORIGINAL verification
scripts at repository revision `49aaee56e082a4b961d979d1a50cfb0d8ae7499c`,
relocated here when the audited exact replacements landed in `scripts/`.

They are retained as historical evidence only. Several of their values are
known to be wrong; the corrected, exact certificates live in
`results/*.json` and are produced by `scripts/run_verifiers.py --write`.
Known corrections (full analysis in the 2026-10-06 audit):

- `gamma70_fusion_results.json`: X(70) cusps/genus/dim M2 are 2x too large
  (3456/18433/21888; correct 1728/9217/10944, projective index mu/2),
  `sin_pi_14_in_SL27_characters: true` is wrong (SL2(7) character field is
  Q(sqrt 2, sqrt -7)), and the 2I Frobenius-Schur list is mislabeled for 4R
  and 6 even though the 96/93/108 aggregates happen to be right.
- `verify_2I_reps_results.json`: floating-point (not exact) FS indicators;
  the spinor search ignored its own `valid_s`/`valid_c` flags.
- `eclectic_nonsplit_results.json`: hard-coded negative verdicts instead of
  computations; the S3 x C4 non-split extension counterexample contradicts
  the blanket Schur-multiplier claim; the "Q(zeta_60) at most" twisted-center
  cap is refuted by the C2 cocycle example; pi2 = SL2(7) is not valid
  2-group data.
- `compactification_search_results.json`: product real dimensions are wrong
  (e.g. S^3/2I x T^2 listed as 4; it is 5), and catalog omissions were
  reported as nonexistence proofs.
- `supplement_verify_output.txt`: never-failing check harness (exit 0 even
  for failing checks; `assert`-based checks disappear under `python -O`).

The generating scripts for these outputs are the pre-replacement versions,
preserved in Git history at the revision above.
