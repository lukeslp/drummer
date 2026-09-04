# Drummer contextual-omission dataset

Author: Luke Steuber. Original synthetic data: CC BY 4.0.

## Generation and purpose

Procedural referential episodes over 64 attribute identities. A sender chooses
whether and what to communicate; a receiver selects one of four candidates.
Grounding delivery and acknowledged history determine whether omission is useful.

Default sizes are 100,000 train, 10,000 validation, and 10,000 sealed test.
The configured mixture is 60% valid repetition, 20% repetition after dropped
grounding, and 20% new target. Exact achieved counts, identity balance, split
membership, generator version, seeds, and SHA-256 hashes belong in the manifest.

## Data boundaries

Public task labels and simulator state are evaluation/training supervision, never
part of the sender or gloss observation. Test access requires explicit unsealing
after freezing model selection. This is an experiment-integrity control, not
cryptographic protection against a person who can regenerate the generator.

No private conversations, credentials, clinical material, or third-party images
are included. Historical plans are documentation, not training data. External
datasets require separate license review before ingestion.

## Generalization

Scene/transition groups are split; all 64 identities may be observed in training.
Results do not establish unseen-identity or compositional generalization. Correlated
episodes are not independent training seeds. Report per-seed uncertainty and all
constructed diagnostic strata.
