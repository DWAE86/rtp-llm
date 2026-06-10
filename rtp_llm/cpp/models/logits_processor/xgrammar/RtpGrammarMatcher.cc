#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/StreamGrammarState.h"

#include <stdexcept>
#include <utility>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RtpGrammarMatcher::RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                     bool                                       require_reasoning,
                                     std::optional<int32_t>                     think_end_id,
                                     std::optional<int32_t>                     think_start_id,
                                     std::optional<std::vector<int32_t>>        override_stop_tokens,
                                     int                                        max_rollback_tokens):
    compiled_(std::move(compiled)),
    // Reasoner mode without a resolved think_end_id can never transition out
    // of the <think> body — isPassthroughForMask() would stay true forever
    // and the grammar would never apply. Degrade to plain grammar mode (no
    // <think> passthrough) instead of silently producing unconstrained
    // output. Previous code asserted on this, but assert is a no-op in
    // release builds, so the bug shipped as silent passthrough.
    require_reasoning_(require_reasoning && think_end_id.has_value()),
    think_end_id_(think_end_id),
    think_start_id_(think_start_id),
    override_stop_tokens_(override_stop_tokens) {
    if (require_reasoning && !think_end_id.has_value()) {
        RTP_LLM_LOG_WARNING(
            "RtpGrammarMatcher: reasoning requested but think_end_id missing; "
            "falling back to plain grammar (no <think> passthrough phase)");
    }
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }

    max_rollback_tokens_ = max_rollback_tokens;

    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(
        *compiled_,
        std::move(override_stop_tokens),
        /*terminate_without_stop_token=*/false,
        max_rollback_tokens);

    // Default: assume request starts in fully-constrained mode unless
    // initReasoning(true) is called by the caller. This matches Python's
    // BaseGrammarObject default where tokens_after_think_end starts at 0.
    tokens_after_think_end_ = 0;
}

void RtpGrammarMatcher::initReasoning(bool in_think_body) {
    if (!require_reasoning_) {
        // Phase counter is never read in non-reasoning mode; leave at 0.
        return;
    }
    tokens_after_think_end_ = in_think_body ? -1 : 0;
}

bool RtpGrammarMatcher::acceptToken(int32_t token_id) {
    ++stats_.accept_calls;

    // In reasoning passthrough phase the parser is intentionally frozen.
    // We DO update the reasoner counter so that the eventual think_end_id
    // is observed — but the grammar matcher itself is untouched.
    if (isPassthroughForMask()) {
        transferReasonerState(token_id);
        ++num_accepted_;
        accepted_history_.push_back(token_id);
        if (static_cast<int>(accepted_history_.size()) > max_rollback_tokens_) {
            accepted_history_.pop_front();
        }
        return true;
    }

    // Active phase: ask the parser. xgrammar may return false for an illegal
    // token without throwing — the caller must check and decide. Do NOT
    // assert here; replay/sampler paths have legitimate "false → handle
    // gracefully" semantics.
    const bool ok = matcher_->AcceptToken(token_id);
    if (!ok) {
        ++stats_.accept_failures;
        RTP_LLM_LOG_WARNING(
            "RtpGrammarMatcher::acceptToken REJECTED token=%d, num_accepted=%ld, "
            "tokens_after_think_end=%d, require_reasoning=%d, terminated=%d",
            token_id, num_accepted_, tokens_after_think_end_,
            static_cast<int>(require_reasoning_),
            static_cast<int>(matcher_->IsTerminated()));
        return false;
    }
    transferReasonerState(token_id);
    ++num_accepted_;
    accepted_history_.push_back(token_id);
    if (static_cast<int>(accepted_history_.size()) > max_rollback_tokens_) {
        accepted_history_.pop_front();
    }
    return true;
}

bool RtpGrammarMatcher::acceptTokens(const std::vector<int32_t>& tokens) {
    for (int32_t t : tokens) {
        if (!acceptToken(t)) {
            return false;
        }
    }
    return true;
}

bool RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) {
    if (isPassthroughForMask()) {
        // Leave the bitmask row at the caller-provided default (all-allow).
        // Returning false signals "no apply needed" so the kernel can skip.
        return false;
    }
    const bool wrote_mask = matcher_->FillNextTokenBitmask(bitmask, idx);
    if (wrote_mask) {
        ++stats_.mask_apply_count;
    }
    return wrote_mask;
}

bool RtpGrammarMatcher::isTerminated() const {
    return matcher_->IsTerminated();
}

void RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return;
    }
    ++stats_.rollback_calls;
    // Compute how many of the rollback steps fall in the active (parser-
    // advancing) phase vs the passthrough phase. Only active-phase steps
    // touch the parser; passthrough steps only walk the reasoner counter
    // backwards.
    //
    // In non-reasoning mode, `tokens_after_think_end_` is never incremented
    // (transferReasonerState early-returns), so every accepted token already
    // went through the parser — active_steps = n. Falling back to the
    // reasoning-mode formula in that case would clamp to 0 and silently
    // skip matcher_->Rollback(), leaving xgrammar's parser advanced past
    // tokens the caller thinks have been rolled back. That de-syncs the
    // matcher state and is invisible in single-token decode (no one calls
    // rollback) but catastrophic under MTP: every spec step's DFS walk
    // drifts the parser forward without rolling back, so the next bitmask
    // is computed against a stale state and allows non-compliant tokens.
    const int active_steps =
        require_reasoning_ ? std::min<int>(n, std::max<int>(0, tokens_after_think_end_)) : n;
    if (active_steps > 0) {
        matcher_->Rollback(active_steps);
    }
    for (int i = 0; i < n; ++i) {
        rollbackReasonerState();
        if (!accepted_history_.empty()) {
            accepted_history_.pop_back();
        }
    }
    num_accepted_ -= n;
    if (num_accepted_ < 0) {
        num_accepted_ = 0;
    }
}

void RtpGrammarMatcher::transferReasonerState(int32_t token_id) noexcept {
    if (!require_reasoning_) {
        return;
    }
    if (tokens_after_think_end_ < 0) {
        // Currently inside <think>...</think>. The think_end token closes
        // the body; nothing else changes phase.
        if (think_end_id_.has_value() && token_id == *think_end_id_) {
            tokens_after_think_end_ = 0;
        }
        // else: still inside <think>, counter stays at -1.
    } else {
        // Already past thinking — count tokens for diagnostics / rollback.
        ++tokens_after_think_end_;
    }
}

void RtpGrammarMatcher::rollbackReasonerState() noexcept {
    if (!require_reasoning_) {
        return;
    }
    if (tokens_after_think_end_ == 0) {
        // Reverting the transition step: we were just past <think>; now we
        // are back inside the body.
        tokens_after_think_end_ = -1;
    } else if (tokens_after_think_end_ > 0) {
        --tokens_after_think_end_;
    }
    // No-op when already at -1: rolling back a still-thinking token does
    // not change phase (the parser was untouched on accept either).
}

std::unique_ptr<RtpGrammarMatcher> RtpGrammarMatcher::fork() const {
    auto forked = std::make_unique<RtpGrammarMatcher>(
        compiled_, require_reasoning_, think_end_id_, think_start_id_,
        override_stop_tokens_, max_rollback_tokens_);

    if (!require_reasoning_) {
        forked->tokens_after_think_end_ = 0;
    } else if (tokens_after_think_end_ >= 0) {
        // Parent is past think_end (or never entered think). Seed at 0 so
        // the replayed tail accumulates the correct counter via
        // transferReasonerState. If history is not truncated, replay covers
        // the full sequence; if truncated, the evicted prefix included
        // think_end so seeding at 0 is still correct.
        forked->tokens_after_think_end_ = 0;
    } else {
        forked->tokens_after_think_end_ = -1;
    }

    bool replay_ok = true;
    for (int32_t tok : accepted_history_) {
        if (!forked->acceptToken(tok)) {
            RTP_LLM_LOG_WARNING("[grammar_fork] replay failed at token=%d after %ld accepted",
                                tok, forked->num_accepted_);
            forked->markFinished();
            replay_ok = false;
            break;
        }
    }
    forked->stats_ = stats_;
    RTP_LLM_LOG_INFO("[grammar_fork] forked matcher, replayed %zu tokens, num_accepted=%ld, ok=%d",
                     accepted_history_.size(), forked->num_accepted_, static_cast<int>(replay_ok));
    return forked;
}

StreamGrammarState RtpGrammarMatcher::snapshot() const {
    StreamGrammarState state;
    state.num_accepted           = num_accepted_;
    state.tokens_after_think_end = tokens_after_think_end_;
    state.finished               = finished_;
    state.stats                  = stats_;
    return state;
}

void RtpGrammarMatcher::restore(const StreamGrammarState& state) {
    int64_t delta = num_accepted_ - state.num_accepted;
    if (delta > 0) {
        rollback(static_cast<int>(delta));
    }
    tokens_after_think_end_ = state.tokens_after_think_end;
    finished_               = state.finished;
    stats_                  = state.stats;
    RTP_LLM_LOG_INFO("[grammar_restore] restored state: num_accepted=%ld, tokens_after_think_end=%d",
                     num_accepted_, tokens_after_think_end_);
}

}  // namespace rtp_llm
