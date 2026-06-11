#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <xgrammar/compiler.h>
#include <xgrammar/matcher.h>

namespace rtp_llm {

// Adapter over xgrammar::GrammarMatcher with an optional in-stream reasoning
// (think-body) gate. When require_reasoning is true the matcher starts in
// passthrough: acceptToken keeps advancing a KMP scanner over think_end_token_ids
// without forwarding to the xgrammar parser, and fillBitmask returns false so the
// caller can publish an allow-all (PASSTHROUGH) mask. Once the think-end
// sequence matches, the matcher transitions to active grammar mode and behaves
// like the plain matcher. rollback() walks one ReasonerState per token so spec
// verify can fork/rollback across the gate without leaking active accepts.
//
// Threading: one matcher per GenerateStream, touched by one sampler thread per tick.
class RtpGrammarMatcher {
public:
    RtpGrammarMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                      std::optional<std::vector<int32_t>>        override_stop_tokens         = std::nullopt,
                      bool                                       terminate_without_stop_token = false,
                      int                                        max_rollback_tokens          = 200,
                      bool                                       require_reasoning            = false,
                      std::vector<int>                           think_end_token_ids          = {});

    RtpGrammarMatcher(const RtpGrammarMatcher&)            = delete;
    RtpGrammarMatcher& operator=(const RtpGrammarMatcher&) = delete;
    RtpGrammarMatcher(RtpGrammarMatcher&&)                 = default;
    RtpGrammarMatcher& operator=(RtpGrammarMatcher&&)      = default;

    // Set reasoning entry state. in_think_body=true keeps the matcher in
    // passthrough until the think-end sequence is observed; false starts active
    // (no-op when require_reasoning was constructed false). Must be called before
    // any token is accepted; clears rollback history.
    void initReasoning(bool in_think_body);

    [[nodiscard]] bool acceptToken(int32_t token_id);
    [[nodiscard]] bool acceptTokens(const std::vector<int32_t>& tokens);

    bool fillBitmask(DLTensor* bitmask, int32_t idx) const;

    bool isTerminated() const;
    bool onlyStopTokenLegalNext(int32_t stop_token_id) const;
    void rollback(int n);

    int64_t numAcceptedTokens() const noexcept { return num_accepted_; }
    int32_t vocabSize() const { return compiled_->GetTokenizerInfo().GetVocabSize(); }

    void markFinished() noexcept { finished_ = true; }
    bool finished() const noexcept { return finished_; }

    int maxRollbackTokens() const noexcept { return max_rollback_tokens_; }

    // True iff the matcher is in reasoning passthrough — caller should publish
    // an allow-all mask and skip xgrammar bitmask filling for this position.
    bool isPassthroughForMask() const noexcept {
        return require_reasoning_ && tokens_after_think_end_ < 0;
    }

private:
    void   transferReasonerState(int32_t token_id) noexcept;
    void   rollbackReasonerState() noexcept;
    size_t nextThinkEndMatchPos(int32_t token_id) const noexcept;

    struct ReasonerState {
        int    tokens_after_think_end;
        size_t think_end_match_pos;
    };

    std::shared_ptr<xgrammar::CompiledGrammar> compiled_;
    std::unique_ptr<xgrammar::GrammarMatcher>  matcher_;

    int64_t num_accepted_        = 0;
    bool    finished_            = false;
    int     max_rollback_tokens_ = 200;

    // Reasoning gate (no-op when require_reasoning_ is false).
    bool                       require_reasoning_      = false;
    std::vector<int>           think_end_token_ids_;
    std::vector<size_t>        think_end_lps_;
    int                        tokens_after_think_end_ = 0;  // < 0 => inside think body, parser frozen.
    size_t                     think_end_match_pos_    = 0;
    std::vector<ReasonerState> reasoner_state_history_;
};

}  // namespace rtp_llm
