#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace rtp_llm {

struct StreamGrammarState;

// Per-request grammar debug counters. Hot-path counters are mutated by the
// matcher; descriptive fields (compile_time_us, cache_hit, dispatch_type) are
// filled by GrammarLogitsProcessor::prepare when the matcher is installed.
struct GrammarStats {
    int64_t     compile_time_us = 0;       // 0 when served from memory cache
    bool        cache_hit       = false;
    std::string dispatch_type;             // json / regex / ebnf / structural_tag

    int64_t mask_apply_count = 0;
    int64_t accept_calls     = 0;
    int64_t accept_failures  = 0;
    int64_t rollback_calls   = 0;
};

// Adapter over xgrammar::GrammarMatcher.
//   * Pins CompiledGrammar via shared_ptr — matcher must not outlive its
//     parser tables.
//   * Layers the optional reasoning gate: in <think> body the parser is
//     frozen and acceptToken is vacuous; the matcher transitions out when
//     the configured think_end_id is observed.
//
// Threading: belongs to ONE GenerateStream and is touched by ONE thread per
// tick (the sampler for that stream). No internal sync.
//
// Failure model: AcceptToken returns false (no exception) for an illegal
// token in active mode; we propagate the bool unchanged.
class RtpGrammarMatcher {
public:
    // require_reasoning=true requires think_end_id; if missing we degrade to
    // plain mode (with a warning) instead of getting stuck in passthrough.
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      bool                                       require_reasoning,
                      std::optional<int32_t>                     think_end_id,
                      std::optional<int32_t>                     think_start_id       = std::nullopt,
                      std::optional<std::vector<int32_t>>        override_stop_tokens = std::nullopt,
                      int                                        max_rollback_tokens  = 200);

    RtpGrammarMatcher(const RtpGrammarMatcher&)            = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher(RtpGrammarMatcher&&)                 = default;
    RtpGrammarMatcher& operator=(RtpGrammarMatcher&&)      = default;

    // Re-arm or skip the reasoning gate. Pass true if the request currently
    // sits inside <think>.
    void initReasoning(bool in_think_body);

    // Returns false ONLY in active mode when the parser rejected the token.
    // Passthrough phase always returns true (vacuous accept).
    [[nodiscard]] bool acceptToken(int32_t token_id);
    [[nodiscard]] bool acceptTokens(const std::vector<int32_t>& tokens);

    // Returns true iff a non-default mask was written. Passthrough returns
    // false (row stays at the caller's all-allow default).
    bool fillBitmask(DLTensor* bitmask, int32_t idx);

    // True when this matcher should NOT participate in batch mask apply
    // (passthrough during <think>); the row stays all-1s.
    bool isPassthroughForMask() const noexcept {
        return require_reasoning_ && tokens_after_think_end_ < 0;
    }

    bool isTerminated() const;

    // Roll back the last `n` accepted tokens (parser state + reasoning counter).
    void rollback(int n);

    int64_t numAcceptedTokens() const noexcept { return num_accepted_; }
    int32_t vocabSize() const { return compiled_->GetTokenizerInfo().GetVocabSize(); }

    // Resolved think-end token id (if reasoner-capable). installMatcher uses
    // this to scan the prompt and pick the correct initial passthrough/active
    // state independent of any per-request server flag.
    std::optional<int32_t> thinkEndId() const noexcept { return think_end_id_; }
    std::optional<int32_t> thinkStartId() const noexcept { return think_start_id_; }
    bool                   reasonerCapable() const noexcept { return require_reasoning_; }

    void markFinished() noexcept { finished_ = true; }
    bool finished() const noexcept { return finished_; }

    const GrammarStats& stats() const noexcept        { return stats_; }
    GrammarStats&       mutableStats() noexcept       { return stats_; }

    std::unique_ptr<RtpGrammarMatcher> fork() const;
    StreamGrammarState snapshot() const;
    void restore(const StreamGrammarState& state);

    int maxRollbackTokens() const noexcept { return max_rollback_tokens_; }

private:
    // Called AFTER the parser has been (or chosen not to be) advanced.
    void transferReasonerState(int32_t token_id) noexcept;
    void rollbackReasonerState() noexcept;

    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    const bool                              require_reasoning_;
    const std::optional<int32_t>            think_end_id_;
    const std::optional<int32_t>            think_start_id_;
    const std::optional<std::vector<int32_t>> override_stop_tokens_;

    // < 0 : inside <think> (passthrough; parser frozen).
    // == 0: just consumed think_end_id (transition step).
    // > 0 : active mode; value = tokens accepted since exit-from-thinking.
    // Stays at 0 forever when require_reasoning_ is false ("always active").
    int tokens_after_think_end_ = 0;

    int64_t              num_accepted_ = 0;
    bool                 finished_     = false;
    GrammarStats         stats_;
    int                  max_rollback_tokens_ = 200;
    std::deque<int32_t>  accepted_history_;
};

}  // namespace rtp_llm
