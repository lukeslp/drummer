"""Offline closed-loop rewriting instrument; no model, file or network authority.

Candidates establish actual recipient state before scoring. Expected meanings
are read only by the report scorer, never by a rewriter, fallback or ACK policy.
This module does not certify arbitrary English or execute requested actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

from drummer.rewrite_corpus import RECIPIENT, RewriteConversation, render_rule, render_terse
from drummer.rewrite_semantics import ParsedMessage, compare_meanings, parse_message
from drummer.rewrite_state import RewriteLedger


EVALUATOR_VERSION = "rewrite-evaluation-v1"
_STATUSES = ("complete", "decode_budget_exhausted", "time_budget_exhausted", "invalid_input",
             "invalid_output", "nonfinite_logits", "runtime_error")


@dataclass(frozen=True)
class RewriteAttempt:
    """Expanded candidate, not a COPY-token stream. Partial text is charged too."""

    status: str
    text: str | None = None
    internal_output_tokens: int | None = None
    error: str | None = None

    def __post_init__(self):
        if self.status not in _STATUSES or type(self.status) is not str:
            raise ValueError("unknown generation status")
        if self.text is not None:
            if type(self.text) is not str:
                raise ValueError("candidate must be text or None")
            self.text.encode("utf-8", errors="strict")
        if self.status == "complete" and (self.text is None or self.error is not None):
            raise ValueError("complete attempt requires text and no generation error")
        if self.internal_output_tokens is not None and (
                type(self.internal_output_tokens) is not int or self.internal_output_tokens < 0):
            raise ValueError("internal output token count must be nonnegative or unavailable")
        if self.error is not None and type(self.error) is not str:
            raise ValueError("error must be text or None")


Rewriter = Callable[[str, str], RewriteAttempt]


def _bytes(text):
    return len(text.encode("utf-8")) if text is not None else 0


def evaluate_conversation(conversation: RewriteConversation, rewriter: Rewriter | str, *,
                          fallback: bool = False) -> dict:
    """Run one actual-output history with one attempt and at most one fallback/turn.

    Neural callbacks receive only source English and visible recipient context.
    Full/terse/rule are authored controls, selected by name. All controls parse
    source text, not gold records. Fallback responds only to a parse/generation
    failure or explicit abstention, never to a semantic score or a transport drop.
    Valid-but-wrong output remains wrong and changes state as actually delivered.
    """
    if type(conversation) is not RewriteConversation or type(fallback) is not bool:
        raise ValueError("validated conversation and boolean fallback are required")
    if not callable(rewriter) and rewriter not in ("full", "terse", "rule"):
        raise ValueError("rewriter must be a callback or a registered authored baseline")
    baseline = rewriter if type(rewriter) is str else None
    ledger = RewriteLedger()
    rows = []
    began = time.monotonic()
    for turn in conversation.turns:
        controls = [ledger.restart(RECIPIENT)] if turn.reset_before else []
        context = ledger.visible_context(RECIPIENT)
        snapshot = ledger.snapshot(RECIPIENT)
        started = time.monotonic()
        try:
            if baseline is None:
                attempt = rewriter(turn.source, context)
                if type(attempt) is not RewriteAttempt:
                    raise ValueError("rewriter did not return an exact RewriteAttempt")
            elif baseline == "full":
                attempt = RewriteAttempt("complete", turn.source)
            else:
                source_parse = parse_message(turn.source)
                if source_parse.error or source_parse.meaning is None:
                    attempt = RewriteAttempt("invalid_input", error="source is outside the authored grammar")
                else:
                    text = (render_rule(source_parse.meaning, snapshot) if baseline == "rule"
                            else render_terse(source_parse.meaning))
                    attempt = RewriteAttempt("complete", text)
        except Exception as error:
            # Unexpected errors remain failed attempts. No retries or gold-based
            # recovery; emitted partial output is unknown if the callback raised.
            attempt = RewriteAttempt("runtime_error", error=f"{type(error).__name__}: {error}")
        generation_seconds = time.monotonic() - started
        candidate = (parse_message(attempt.text, snapshot) if attempt.status == "complete"
                     else ParsedMessage(None, error=attempt.status))
        delivered_text = attempt.text if attempt.status == "complete" else None
        events = []
        if delivered_text is not None:
            events.append(ledger.receive(RECIPIENT, candidate, payload_delivered=turn.payload_delivered,
                                         ack_delivered=turn.ack_delivered))
        first_received = candidate if turn.payload_delivered else ParsedMessage(None, error="payload dropped")
        final_received = first_received
        fallback_text = None
        # This guard sees parser/generation outcomes, not compare_meanings.
        if fallback and (candidate.error is not None or candidate.abstained):
            fallback_text = turn.source
            fallback_parse = parse_message(fallback_text, ledger.snapshot(RECIPIENT))
            events.append(ledger.receive(RECIPIENT, fallback_parse,
                                         payload_delivered=turn.payload_delivered,
                                         ack_delivered=turn.ack_delivered))
            final_received = (fallback_parse if turn.payload_delivered
                              else ParsedMessage(None, error="payload dropped"))
        # No expected record has been accessed before actual deliveries/state.
        first_score = compare_meanings(turn.expected, first_received)
        final_score = compare_meanings(turn.expected, final_received)
        candidate_score = compare_meanings(turn.expected, candidate)
        forward = _bytes(delivered_text) + _bytes(fallback_text) + sum(e.forward_utf8_bytes for e in controls)
        feedback = sum(event.feedback_utf8_bytes for event in events)
        source_bytes, context_bytes, proposal_bytes = _bytes(turn.source), _bytes(context), _bytes(attempt.text)
        rows.append({
            "event": turn.event, "attempt": asdict(attempt), "generation_seconds": generation_seconds,
            "candidate_score": candidate_score, "first_pass_score": first_score, "final_score": final_score,
            "fallback_used": fallback_text is not None, "fallback_text": fallback_text,
            "reference_only": candidate.reference_only, "abstained": candidate.abstained,
            "state_events": [asdict(event) for event in controls + events],
            "state_after": [asdict(binding) for binding in ledger.snapshot(RECIPIENT)],
            "context_before": context,
            "bytes": {"rewriter_source": source_bytes, "rewriter_context": context_bytes,
                      "rewriter_output_observed": proposal_bytes,
                      "recipient_forward_emitted": forward, "coordinator_feedback_emitted": feedback,
                      "modeled_text_total": source_bytes + context_bytes + proposal_bytes + forward + feedback},
            "output_accounting_complete": attempt.status != "runtime_error",
        })
    count = len(rows)
    totals = {key: sum(row["bytes"][key] for row in rows) for key in rows[0]["bytes"]}
    return {
        "format": EVALUATOR_VERSION, "conversation_id": conversation.conversation_id,
        "policy_kind": "authored_baseline" if baseline else "callback", "baseline": baseline,
        "fallback_enabled": fallback, "turns": rows,
        "summary": {
            "turns": count,
            "candidate_joint": sum(row["candidate_score"]["joint"] for row in rows),
            "first_pass_joint": sum(row["first_pass_score"]["joint"] for row in rows),
            "final_joint": sum(row["final_score"]["joint"] for row in rows),
            "fallbacks": sum(row["fallback_used"] for row in rows),
            "abstentions": sum(row["abstained"] for row in rows),
            "generation_failures": sum(row["attempt"]["status"] != "complete" for row in rows),
            "unassisted_semantic_coverage": sum(row["first_pass_score"]["joint"] and not row["fallback_used"]
                                                for row in rows) / count,
            "bytes": totals, "elapsed_seconds": time.monotonic() - began,
            "output_accounting_complete": all(row["output_accounting_complete"] for row in rows),
        },
        "native_endpoint_tokens": None, "provider_cost_usd": None, "training_cost_usd": None,
        "executed_actions": 0, "training_started": False, "promotion_eligible": False,
        "scope": "offline instrument; UTF-8 boundaries are not provider token usage or a savings claim",
    }
