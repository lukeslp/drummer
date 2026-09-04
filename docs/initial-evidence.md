# Initial evidence — 2026-09-04

Author: Luke Steuber. Status: implementation validation and exploratory results,
not a promoted learned language or a deployment recommendation.

All runs below used source revision
`949d5be04729b9aa2e5e93ea5e9fa7a90370d155`. The dependency-lock SHA-256 was
`b41960a0339fccf6893ff20a24dd578875d4e2368e624c8f13b4d8f2641680bd`.
The exact clean revision passed 149 tests, Ruff, documentation drift checks,
and [hosted verification](https://github.com/lukeslp/drummer/actions/runs/33916520331).

## Learned-channel correctness

The default 3,430,977-parameter architecture ran locally on MPS and on one NVIDIA
L4. The CUDA smoke completed 16 optimizer steps, uploaded checkpoints, and
reported a finite best validation loss of 1.390943. Its measured worker interval
was 9.282 seconds, including corpus setup and artifact work. This tiny run is
not a reliable estimate of full-corpus throughput. The underlying training
report is explicitly partial because it stopped at the smoke's step limit.

CUDA used Python 3.12.14, PyTorch 2.14.0+cu130, CUDA 13.0, and cuDNN 92400.
The container and dependency-tool digests are in the immutable run manifest.
Both MPS and CUDA emitted nondeterministic-operation warnings despite requesting
deterministic algorithms in warning-only mode. Seeds and runtime identifiers
support reproducibility attempts; bitwise-identical replay is not established.

On the 80-example validation smoke, the full-description and deterministic
contextual-omission controls both achieved 100% success. The latter used 3.4
forward bits against six for compulsory identity transmission: 43.3% fewer
forward bits. Historical grounding and acknowledgement add seven bits in these
fixtures. This is the known hand-written rule, not a learned result.

The 100,000/10,000/10,000 pilot corpus was generated with format-v3 integrity
manifests. Its test remains sealed during calibration. No five-seed promotion
or unfamiliar-partner result has been established by this report.

## Real coding-client handoffs

The frozen corpus is `synthetic-24-v2`; the response contract is
`ordered-process-steps-v2`. Each direction used the same 24 cases, with no tools,
production edits, automatic repairs, or metered API fallback. The sender's
actual output, not a substituted reference answer, reached the receiver.

Codex CLI 0.153.2 requested `gpt-6-astra`. Claude Code 2.1.258 used its installed
default; its usage report named `claude-opus-5[1m]` and an auxiliary
`claude-haiku-4-5-20251001`. These are client-level measurements, not a claim that
one isolated Claude checkpoint handled every internal operation.

| Direction / format | Valid senders | Strict exact responses | Total input + output tokens | Mean elapsed seconds |
|---|---:|---:|---:|---:|
| Codex → Claude / full English | 24/24 | 4/24 | 473,003 | 23.73 |
| Codex → Claude / terse English | 22/24 | 5/24 | 451,803 | 16.06 |
| Claude → Codex / full English | 24/24 | 21/24 | 507,132 | 24.49 |
| Claude → Codex / terse English | 24/24 | 17/24 | 482,559 | 16.81 |

All 20 strict failures in the full-English Codex → Claude direction failed JSON
parsing. In the reverse full-English direction, three responses changed the case identifier while preserving
the scored step values. Formatting and identifier errors remain failures under
the frozen contract. This does not establish an inability to understand the
underlying English, and the primary scores have not been relaxed after seeing
the outputs.

Tokens include both endpoints' complete reported input and output, including
Claude cache creation and cache reads. Missing auxiliary metrics remain missing.
Elapsed times include live client overhead and concurrent work; they are not
controlled isolated latency benchmarks. Terse English used about 4.5% fewer
tokens in Codex → Claude and 4.8% fewer in Claude → Codex across these attempts.
The former included two sender rejections with no receiver call; the latter
lost four exact successes. These totals are not an equivalent-quality savings
claim. The full ablation matrix remains uncompleted.

The [machine-readable measurement extract](evidence/2026-09-04.json) contains
per-attempt scores, token counts, elapsed times, contract hashes where applicable,
and SHA-256 fingerprints of the preserved raw records. It covers all 96 paired
prose attempts, the four packet sender probes, and nine local receiver attempts
reported here. Earlier one-case instrumentation probes remain separate records.

Two one-case native-packet sender probes were run per direction: one with the
generic JSON schema and one with the normative manual as the encoding contract.
All four were rejected before receiver delivery. Recorded failures included
authority data placed inside a packet, invalid textual reference order, schema
violations, and non-JSON wrapping. These are failed end-to-end protocol probes,
not four receiver-comprehension observations. Neither contract included the
case-specific gold response, and no oracle packet was substituted.

## Small-model transfer smoke

Each cell below is one attempt on `process_ambiguity-1`, not a 24-case estimate.
All three representations used the same response contract and legitimate
meaning. Native and expanded protocol inputs came from deterministic fixtures;
these are one-receiver tests, not learned model-to-model packet production.

| Installed model | Full English | Native packet | Deterministic expansion |
|---|---|---|---|
| Qwen2.5 0.5B | Fail; 895 tokens | Fail; 989 tokens | Fail; 713 tokens |
| Qwen2.5 1.5B | Fail; 891 tokens | Fail; 938 tokens | Fail; 716 tokens |
| Qwen3 8B | Pass; 1,280 tokens | Fail; 1,588 tokens | Fail; 1,337 tokens |

Tokens are complete reported input plus output for that attempt. A shorter
failed response is not a saving at equivalent task quality. The 0.5B and 1.5B
responses included JSON wrapping and identifier/field errors. The 8B protocol
responses parsed but changed a required value. Deterministic English expansion
therefore did not establish compatibility in these particular probes.

The smaller two models used Ollama 0.32.3; the 8B model used Ollama 0.33.3.
All were installed Q4_K_M GGUF models. Temperature was zero and maximum output
was 2,048 tokens. No model was downloaded for the experiment.

| Model | Installed artifact digest |
|---|---|
| Qwen2.5 0.5B | `a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67` |
| Qwen2.5 1.5B | `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b` |
| Qwen3 8B | `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` |

## Interpretation and next decision

The useful result is a working, instrumented experiment with visible failure
boundaries. The initial evidence does not support claiming that canonical JSON
is already an efficient language, that these local models understand Drummer
natively, or that reduced character counts yield lower end-to-end costs.

The next learned run is bounded calibration followed by the preregistered pilot
only if validation quality permits. Protocol work should separately test a
versioned, smaller encoding/decoding contract and an explicit repair policy,
retaining first-pass scores. No discovered convention has been promoted to the
language atlas or silently assigned a new meaning.
