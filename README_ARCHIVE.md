# MECoFlow / CAMS Research Archive

Archive directory: `./MECoFlow_CAMS_archive_20260825`

This archive is a copy of the research products generated for the MECoFlow/CAMS
compositional zero-shot learning study. The original working tree at
`./CAMS` was left unchanged.

## Contents

- `research/`: experiment scripts, frozen protocols, validation/test audit JSON,
  literature comparison, novelty audit, method draft, paper plan, abstract,
  experiment log, completion audit, and generated paper result tables.
- `output/research/`: cached features, flow checkpoints, generated classifier
  logits, source replays, official FlowComposer runs, validation-only probes,
  fixed-test results, and provenance artifacts.
- `output/`: also contains the original CAMS baseline checkpoints, seed runs,
  training logs, and summary CSV files outside `output/research`.
- `code_snapshot/`: CAMS source, configs, bundled external code, feasibility
  tensors, and base weights as archived at completion. `.git`, editor metadata,
  runtime bytecode, and the separately archived `output/` and `research/`
  directories are intentionally excluded from this snapshot.
- `README.md` and `LITERATURE_NOVELTY_AUDIT.md`: project-level source files.
- `ARCHIVE_MANIFEST.txt`: relative file list and byte sizes.
- `ARCHIVE_SHA256SUMS.txt`: SHA256 checksums for every archived file.

## Key verified results

- UT seed-1 candidate-conditioned route: `+8.360 AUC / +11.863 HM` points.
- Original UT candidate-conditioned route: `+8.031 AUC / +11.200 HM` points.
- MIT seed-1 candidate-conditioned route: `+3.381 AUC / +9.025 HM` points.
- C-GQA strict three-source replay: `+1.222 AUC / +5.942 HM` points.

All fixed candidates were selected from validation-only evidence and the final
consistency verifier reported `All consistency assertions passed.` C-GQA raw
distribution completion remains validation-gated; the archive does not claim
absolute cross-paper SOTA.

Archive created: 2026-08-25 (Asia/Shanghai).
