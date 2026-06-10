#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorModules.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarCompiler.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarLogitsProcessor.h"

namespace rtp_llm {

// Definition of the neutral hook declared in LogitsProcessorModules.h. Keeping it
// here is what confines the concrete grammar types to the xgrammar module: the
// engine sees only registerStructuredOutputLogitsProcessor and never includes
// GrammarCompiler / GrammarLogitsProcessor.
void registerStructuredOutputLogitsProcessor(const StructuredOutputConfig& cfg) {
    GrammarCompiler::initialize(cfg);
    LogitsProcessorFactory::registerExtraFactory(&GrammarLogitsProcessor::tryCreatePending);
}

}  // namespace rtp_llm
