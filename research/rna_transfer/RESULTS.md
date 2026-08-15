# RNA-to-protein transfer experiment — 2026-08-13

Status: completed three-seed discovery plus untouched-fold confirmation;
RNA transfer validated as an additive response signal, but not promoted as a
complete model.

This is an independent open-knowledge experiment. It uses the frozen S1
chemical-held-out assignments only for final comparison and does not import,
train, or ensemble the existing GOAI model implementations.

## Locked protocol

- 5,078 treatment samples, 37 held-out chemicals, four folds, fold seed 42.
- 4,422 train-supported proteins; log2 treatment-minus-exact-control response.
- L1000FWD primary source: 38,948 signatures, 4,861 perturbation IDs and 4,276
  standardized parent structures after excluding every GOAI parent match.
- Open Problems secondary source: 598 `logFC` rows and 142 compounds after the
  same exclusion.
- Identical independent downstream protein-delta network for structure-only,
  real-RNA and whole-drug shuffled-RNA arms; model seeds 42, 43 and 2026 for
  the final residual architecture.
- Metrics are local proxies. Fold means are unweighted; confidence intervals
  use 10,000 paired resamples of all rows/proteins within held-out chemicals.

## Four-fold discovery result

| arm | FC PCC | context residual PCC | high-effect PCC | high-effect F1 |
|---|---:|---:|---:|---:|
| no chemical input | **0.429196** | **0.131844** | **0.541367** | 0.107253 |
| Morgan scratch | 0.334650 | 0.122943 | 0.470212 | 0.108172 |
| Open Problems real | 0.362599 | 0.131629 | 0.504149 | 0.113014 |
| Open Problems shuffled | 0.309715 | 0.086713 | 0.478143 | 0.109030 |
| L1000 real, minus GOAI | 0.362795 | 0.117779 | 0.507710 | 0.114884 |
| L1000 shuffled, minus GOAI | 0.351651 | 0.092366 | 0.493914 | **0.120598** |
| existing M6.11, three-seed absolute model | 0.371674 | 0.098239 | **0.635453** | **0.182438** |

Delta-only arms do not claim absolute sample R2: reconstructing absolute
predictions from observed validation controls would be an oracle protocol.

## Attribution tests

- Open Problems real minus shuffled: FC +0.052884, 95% CI
  [0.022465, 0.076849]; context +0.044916 [0.008911, 0.093330]; high-effect
  PCC +0.026005 [0.010730, 0.039794]. This is positive transferable
  representation evidence.
- L1000 real minus shuffled: FC +0.011143 [-0.006808, 0.029744]; context
  +0.025413 [-0.006564, 0.063327]. Direction is positive but uncertainty
  crosses zero.
- L1000 real minus no-chemical: FC -0.066401 [-0.085359, -0.044784] and
  high-effect PCC -0.033657 [-0.060328, -0.014136]. The transfer model does
  not beat the strongest independent consumer.
- Correct-vs-deranged held-out chemical fingerprints, using each same fitted
  L1000 real model: mean FC +0.012020, context +0.006904 and high-effect PCC
  +0.019234. The network uses the drug representation, but the benefit is not
  stable enough to pass the promotion gate.
- Replaying the same saved Open Problems real models with a whole-drug
  derangement gives correct-minus-permuted FC +0.021591, context +0.026970,
  high-effect PCC +0.022970 and high-effect F1 +0.010453. This is the clearest
  evidence that the real RNA encoder contributes drug-specific information.

## Architecture iteration: frozen context + RNA residual

The direct chemical consumer above displaced a stronger context predictor. The
second architecture therefore freezes the context-only model and trains only a
zero-initialized, frozen-RNA chemical residual. A fixed residual scale of 0.20
was selected on S1 fold 0 with seed 42; folds 1–3 were then treated as an
untouched confirmation subset. No outer validation was opened.

Three-seed equal-weight four-fold OOF:

| arm | FC PCC | context residual PCC | high-effect PCC | high-effect F1 |
|---|---:|---:|---:|---:|
| context-only bag | 0.435627 | 0.136079 | 0.544178 | 0.106032 |
| shuffled-RNA residual bag | 0.427173 | 0.123558 | 0.546507 | 0.108425 |
| **real-RNA residual bag** | **0.437656** | **0.139366** | **0.551712** | **0.108180** |
| existing M6.11 absolute model | 0.371674 | 0.098239 | **0.635453** | **0.182438** |

The real residual improved its context-only parent in all three individual
seeds. For the three-seed bag, real minus context-only was FC +0.002029,
context +0.003287, high-effect PCC +0.007534 and high-effect F1 +0.002148.
Chemical-cluster bootstrap CIs crossed zero for FC/context, but were positive
for high-effect PCC [+0.002466, +0.012723] and F1 [+0.001086, +0.003430].

Real minus shuffled was stronger evidence of RNA-specific information: FC
+0.010483 [95% CI +0.004429, +0.016535] and context +0.015808
[+0.004706, +0.035921]. Each of the three seeds had the same positive direction.

On untouched folds 1–3 (28 chemicals), the three-seed real residual versus
context-only changed FC by +0.001322 [-0.002775, +0.006524], context by
+0.001637 [-0.004980, +0.018896], high-effect PCC by +0.006500
[+0.001538, +0.012475], and F1 by +0.002060 [+0.000760, +0.003617]. Thus the
additive high-effect improvement survives removal of the design fold; FC and
context are directionally positive but not yet statistically resolved.

## Decision

The biological premise is supported: real RNA pretraining carries transferable,
drug-specific information, and a zero-initialized residual can add it without
replacing the strong context backbone. The quoted two-metric target is met on
the local S1 proxy: FC 0.437656 and context 0.139366 both exceed M6.11's
0.371674 and 0.098239.

This is still not a complete-model promotion. The new family predicts response
deltas rather than absolute abundance, has no valid absolute-fidelity score,
and remains far below M6.11 on high-effect PCC/F1. It is retained as a promising
open-knowledge response component; outer validation, test inference, blending
with old models, and official submission remain untouched.

The workspace is not a Git repository. L1000 fold manifests capture the source
hashes at process start; earlier OP3/control folds predate that mechanism, so
their configuration, checkpoints and predictions are retained but their exact
runtime source snapshot is a documented reproducibility limitation.

The next honest step is to add an independently predicted absolute background
and a high-effect specialist while keeping this response residual frozen. Only
then can the full metric vector be compared for promotion.
