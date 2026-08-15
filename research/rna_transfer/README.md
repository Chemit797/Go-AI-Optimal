# GOAI RNA-to-protein transfer lab

This directory is an independent open-knowledge experiment. It does not import,
modify, train, or ensemble the existing GOAI model families. The only shared
contracts are the read-only official input CSVs, the frozen S1 fold assignment,
and the metric definition used for the final comparison.

The experiment asks one narrow question: does a molecular encoder trained to
predict external RNA perturbation responses improve protein fold-change
prediction for entirely held-out chemicals?

The primary scientific arm excludes from RNA pretraining any compound whose
canonical parent structure appears anywhere in GOAI. The all-RNA model is kept
only as a clearly labelled same-drug cross-modal upper bound. This distinction
prevents external RNA labels for a held-out GOAI compound from being mistaken
for zero-shot chemical generalization.

## Experimental arms

- `morgan_scratch`: the chemical encoder starts randomly and learns only from
  the GOAI training side of each S1 fold.
- `l1000_real_minus_goai_ft`: identical encoder initialized from 38,948
  L1000FWD signatures after excluding every GOAI parent structure, then
  fine-tuned inside each GOAI fold. This is the primary transfer arm.
- `l1000_shuffle_minus_goai_ft`: identical encoder initialized from a
  parent-structure-shuffled L1000FWD negative control.
- `op3_real_strict_ft` and `op3_shuffle_strict_ft`: smaller Open Problems
  `logFC` domain-adaptation proof and its compound-shuffled control.
- `op3_real_residual_s02` and `op3_shuffle_residual_s02`: final architecture;
  freeze the context-only consumer and add a zero-initialized RNA chemical
  residual at scale 0.20. These are run at seeds 42, 43 and 2026.
- Other `*_shuffle*` arms use a compound-shuffled RNA
  negative control, then fine-tuned identically.
- `rna_real_frozen` and `rna_shuffle_frozen`: stricter representation-only
  controls with the encoder frozen.

Every arm uses the same downstream protein-delta model, protein targets,
chemical-held-out folds, seeds, optimizer, and metric code. RNA pretraining
uses `logFC`, not the original Kaggle signed significance target. External data
and generated artifacts are open-knowledge-only and must not be described as a
closed-data submission.

Heavy public OP3 input should be stored under a local ignored directory such as
`data/external/op3`. GOAI-derived caches belong under the ignored `runs/`
directory and must not be committed.

## Reproduction

Use an isolated CUDA/RDKit scientific environment:

```bash
PY=python

$PY -m goai_rna_transfer.l1000_pretrain verify --data-dir data/external/l1000fwd
$PY -m goai_rna_transfer.l1000_pretrain train --config research/rna_transfer/experiment.local.example.yaml \
  --scope external-minus-goai --label-mode real --device cuda:0 --rank 64
$PY -m goai_rna_transfer.l1000_pretrain train --config research/rna_transfer/experiment.local.example.yaml \
  --scope external-minus-goai --label-mode input-shuffle --device cuda:1 --rank 64
$PY -m goai_rna_transfer.goai_data --config research/rna_transfer/experiment.local.example.yaml
for FOLD in 0 1 2 3; do
  $PY -m goai_rna_transfer.train_s1 --config research/rna_transfer/experiment.local.example.yaml \
    --arm l1000_real_minus_goai_ft --fold "$FOLD" --seed 42 --device cuda:0 \
    --encoder-checkpoint models/l1000_pretraining/l1000_real_minus_goai_encoder.pt
done
$PY -m goai_rna_transfer.evaluate_s1 --config research/rna_transfer/experiment.local.example.yaml \
  --prediction l1000_real=delta:oof/l1000_real_minus_goai_ft/seed_42 \
  --output-dir logs/evaluation/l1000_seed42
```

The OP3 helper must be invoked with `--skip-cv`: its full-data PCA is valid for
final external pretraining but deliberately cannot be reported as an OOF score.

The completed result, including the fold-0 design boundary, three-seed bag and
untouched folds 1–3 confirmation, is in [`RESULTS.md`](RESULTS.md). The persisted
best response OOF is
`oof/ensembles/op3_real_residual_s02_s42_s43_s2026/S1.npz`.

The final promotion gate is paired, not absolute: real RNA pretraining must
beat both scratch Morgan and shuffled-RNA controls on the same OOF rows, with
FC PCC as the primary metric and context-residual PCC plus high-effect recovery
as guardrails.
