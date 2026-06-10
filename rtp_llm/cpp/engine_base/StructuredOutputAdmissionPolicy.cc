#include "rtp_llm/cpp/engine_base/StructuredOutputAdmissionPolicy.h"

#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool isDisabledBackendName(const std::string& name) noexcept {
    return name.empty() || name == "none" || name == "None" || name == "NONE";
}

}  // namespace

bool isStructuredOutputBackendConfigured(const StructuredOutputConfig& engine_cfg) noexcept {
    if (isDisabledBackendName(engine_cfg.grammar_backend)) {
        return false;
    }
    // Only xgrammar is supported today; unknown selectors are treated as disabled
    // (mirrors structured-output module startup wiring).
    if (engine_cfg.grammar_backend != "xgrammar") {
        return false;
    }
    return !engine_cfg.tokenizer_info_json.empty();
}

StructuredOutputAdmissionResult checkStructuredOutputAdmission(const StructuredOutputAdmissionContext& ctx) {
    if (!ctx.has_structured_output_request) {
        return {true, ErrorCode::NONE_ERROR, ""};
    }

    if (!ctx.structured_output_backend_ready) {
        RTP_LLM_LOG_WARNING("[structured_output_admission] rejected: structured output requested but backend not configured");
        return {false,
                ErrorCode::INVALID_PARAMS,
                "structured output requested but constraint backend is not configured or is disabled"};
    }

    if (ctx.num_beams_gt_one) {
        RTP_LLM_LOG_WARNING("[structured_output_admission] rejected: structured output not supported with beam search");
        return {false,
                ErrorCode::INVALID_PARAMS,
                "structured output is not supported together with beam search (num_beams > 1)"};
    }

    if (ctx.num_return_sequences_gt_one) {
        RTP_LLM_LOG_WARNING(
            "[structured_output_admission] rejected: structured output not supported with multiple return sequences");
        return {false,
                ErrorCode::INVALID_PARAMS,
                "structured output is not supported with multiple return sequences (num_return_sequences > 1)"};
    }

    if (ctx.batch_decode_scheduler) {
        RTP_LLM_LOG_WARNING("[structured_output_admission] rejected: structured output not supported with BatchDecodeScheduler");
        return {false,
                ErrorCode::INVALID_PARAMS,
                "structured output is not supported with BatchDecodeScheduler"};
    }

    return {true, ErrorCode::NONE_ERROR, ""};
}

StructuredOutputAdmissionResult
checkStructuredOutputAdmissionForConfig(const std::shared_ptr<GenerateConfig>& config,
                                        const StructuredOutputConfig&          engine_cfg,
                                        bool                                   batch_decode_scheduler) {
    if (!config || !config->hasStructuredOutputRequest()) {
        return {true, ErrorCode::NONE_ERROR, ""};
    }
    StructuredOutputAdmissionContext ctx;
    ctx.has_structured_output_request   = true;
    ctx.structured_output_backend_ready = isStructuredOutputBackendConfigured(engine_cfg);
    ctx.num_beams_gt_one                = config->hasNumBeams();
    ctx.num_return_sequences_gt_one     = config->num_return_sequences > 1;
    ctx.batch_decode_scheduler          = batch_decode_scheduler;
    return checkStructuredOutputAdmission(ctx);
}

}  // namespace rtp_llm
