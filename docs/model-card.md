# Drummer-0 model card

Author: Luke Steuber. License for original weights: Apache-2.0.

## Intended use

Research on contextual omission in a finite synthetic referential world. Not a
general-purpose language model, a production coding agent, or a clinical tool.
Weights start at random initialization; no pretrained language weights are used.

## Architecture

Default: four transformer layers, width 256, four heads, FFN width 1,024,
maximum observation length 128, and an eight-dimensional private residual.
The actual parameter count and device are recorded for each run. Reduced models
used for correctness smoke tests must be labeled as such.

## Evaluation status

No five-seed promotion result has been established. Consult generated run reports
for measured smoke-test or validation results; do not extrapolate them to the
sealed test. No multi-turn or unfamiliar-model transfer claim is currently made.

## Limits

The 64-identity world admits a lookup code and does not test open vocabulary or
compositionality. Its repetition distribution is constructed, not estimated from
real conversation. Shared checkpoint partners are not independent replicas.
Appraisal variables do not establish experienced emotion.

## Release requirements

Publish each selected checkpoint with its safetensors digest, complete config,
training/validation curves, corpus hashes, all five seed results, exact revision,
device/runtime/container details, costs, and limitations. A local smoke checkpoint
may be shared only with its smoke-test label, not as a passed research milestone.
