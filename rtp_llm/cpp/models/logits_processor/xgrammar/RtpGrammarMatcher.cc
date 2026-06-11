#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

// KMP failure-function (longest proper prefix that is also a suffix).
std::vector<size_t> buildKmpFailureTable(const std::vector<int>& pattern) {
    std::vector<size_t> lps(pattern.size(), 0);
    size_t              k = 0;
    for (size_t i = 1; i < pattern.size(); ++i) {
        while (k > 0 && pattern[k] != pattern[i]) {
            k = lps[k - 1];
        }
        if (pattern[k] == pattern[i]) {
            ++k;
        }
        lps[i] = k;
    }
    return lps;
}

}  // namespace

RtpGrammarMatcher::RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                     std::optional<std::vector<int32_t>>        override_stop_tokens,
                                     bool                                       terminate_without_stop_token,
                                     int                                        max_rollback_tokens,
                                     bool                                       require_reasoning,
                                     std::vector<int>                           think_end_token_ids):
    compiled_(std::move(compiled)),
    max_rollback_tokens_(max_rollback_tokens),
    require_reasoning_(require_reasoning),
    think_end_token_ids_(std::move(think_end_token_ids)) {
    if (!compiled_) {
        throw std::invalid_argument("RtpGrammarMatcher requires a non-null CompiledGrammar");
    }
    if (require_reasoning_ && think_end_token_ids_.empty()) {
        throw std::invalid_argument("RtpGrammarMatcher reasoning gate requires non-empty think_end_token_ids");
    }

    matcher_ = std::make_unique<xgrammar::GrammarMatcher>(*compiled_,
                                                          std::move(override_stop_tokens),
                                                          terminate_without_stop_token,
                                                          max_rollback_tokens);

    if (require_reasoning_) {
        think_end_lps_ = buildKmpFailureTable(think_end_token_ids_);
    }
}

void RtpGrammarMatcher::initReasoning(bool in_think_body) {
    if (!require_reasoning_) {
        return;
    }
    tokens_after_think_end_ = in_think_body ? -1 : 0;
    think_end_match_pos_    = 0;
    reasoner_state_history_.clear();
}

bool RtpGrammarMatcher::acceptToken(int32_t token_id) {
    const bool passthrough = isPassthroughForMask();
    if (passthrough) {
        // In think body: do not advance the parser; only step the KMP scanner.
        transferReasonerState(token_id);
        ++num_accepted_;
        return true;
    }

    const bool ok = matcher_->AcceptToken(token_id);
    if (!ok) {
        RTP_LLM_LOG_WARNING("RtpGrammarMatcher::acceptToken REJECTED token=%d, num_accepted=%ld, terminated=%d",
                            token_id,
                            num_accepted_,
                            static_cast<int>(matcher_->IsTerminated()));
        return false;
    }
    transferReasonerState(token_id);
    ++num_accepted_;
    return true;
}

bool RtpGrammarMatcher::acceptTokens(const std::vector<int32_t>& tokens) {
    for (int32_t token_id : tokens) {
        if (!acceptToken(token_id)) {
            return false;
        }
    }
    return true;
}

bool RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) const {
    if (isPassthroughForMask()) {
        return false;
    }
    return matcher_->FillNextTokenBitmask(bitmask, idx);
}

bool RtpGrammarMatcher::isTerminated() const {
    return matcher_->IsTerminated();
}

bool RtpGrammarMatcher::onlyStopTokenLegalNext(int32_t stop_token_id) const {
    if (isTerminated() || finished()) {
        return true;
    }
    if (isPassthroughForMask()) {
        return false;
    }
    const int32_t grammar_vocab_size = vocabSize();
    if (grammar_vocab_size <= 0) {
        return false;
    }
    const size_t words = static_cast<size_t>((grammar_vocab_size + 31) / 32);
    std::vector<int32_t> bitmask(words, 0);
    int64_t              shape[2] = {1, static_cast<int64_t>(words)};
    DLTensor             dl;
    dl.data        = bitmask.data();
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    dl.shape       = shape;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    if (!fillBitmask(&dl, 0)) {
        return false;
    }
    for (int32_t token_id = 0; token_id < grammar_vocab_size; ++token_id) {
        if (token_id == stop_token_id) {
            continue;
        }
        const int32_t word = bitmask[static_cast<size_t>(token_id) / 32];
        if ((static_cast<uint32_t>(word) & (1u << (token_id % 32))) != 0u) {
            return false;
        }
    }
    if (stop_token_id < 0 || stop_token_id >= grammar_vocab_size) {
        return false;
    }
    const int32_t word = bitmask[static_cast<size_t>(stop_token_id) / 32];
    return (static_cast<uint32_t>(word) & (1u << (stop_token_id % 32))) != 0u;
}

void RtpGrammarMatcher::rollback(int n) {
    if (n <= 0) {
        return;
    }
    // Active steps = how many of the rolled-back tokens were forwarded to the parser.
    // Reasoning mode: only post-think tokens were forwarded; cap by tokens_after_think_end_.
    // Non-reasoning: every accept was forwarded (this also fixes the previous bug where
    // active_steps was computed as min(n, max(0, tokens_after_think_end_)) for all calls,
    // which yielded 0 for non-reasoning streams and silently dropped matcher rollbacks).
    const int active_steps =
        require_reasoning_ ? std::min(n, std::max(0, tokens_after_think_end_)) : n;
    if (active_steps > 0) {
        matcher_->Rollback(active_steps);
    }
    if (require_reasoning_) {
        for (int i = 0; i < n; ++i) {
            rollbackReasonerState();
        }
    }
    num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
}

void RtpGrammarMatcher::transferReasonerState(int32_t token_id) noexcept {
    if (!require_reasoning_) {
        return;
    }
    reasoner_state_history_.push_back({tokens_after_think_end_, think_end_match_pos_});
    if (tokens_after_think_end_ < 0) {
        // Inside think body: step the KMP scanner.
        think_end_match_pos_ = nextThinkEndMatchPos(token_id);
        if (think_end_match_pos_ == think_end_token_ids_.size()) {
            // Think-end sequence consumed; transition to active grammar.
            tokens_after_think_end_ = 0;
            think_end_match_pos_    = 0;
        }
    } else {
        ++tokens_after_think_end_;
    }
}

void RtpGrammarMatcher::rollbackReasonerState() noexcept {
    if (reasoner_state_history_.empty()) {
        return;
    }
    const ReasonerState prev = reasoner_state_history_.back();
    reasoner_state_history_.pop_back();
    tokens_after_think_end_ = prev.tokens_after_think_end;
    think_end_match_pos_    = prev.think_end_match_pos;
}

size_t RtpGrammarMatcher::nextThinkEndMatchPos(int32_t token_id) const noexcept {
    size_t k = think_end_match_pos_;
    while (k > 0 && think_end_token_ids_[k] != token_id) {
        k = think_end_lps_[k - 1];
    }
    if (think_end_token_ids_[k] == token_id) {
        ++k;
    }
    return k;
}

}  // namespace rtp_llm
