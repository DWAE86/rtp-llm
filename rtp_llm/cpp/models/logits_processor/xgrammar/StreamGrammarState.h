#pragma once

#include <cstdint>
#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"

namespace rtp_llm {

struct StreamGrammarState {
    int64_t      num_accepted           = 0;
    int32_t      tokens_after_think_end = -1;
    int32_t      replayed_token_count   = 0;
    int32_t      pd_bonus_ingested      = 0;
    bool         finished               = false;
    GrammarStats stats;
};

}  // namespace rtp_llm
