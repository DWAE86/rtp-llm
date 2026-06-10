#pragma once

#include <memory>
#include <string>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

class GenerateConfig;

struct StructuredOutputAdmissionContext {
    bool has_structured_output_request   = false;
    bool structured_output_backend_ready = false;
    bool num_beams_gt_one                = false;
    bool num_return_sequences_gt_one     = false;
    bool batch_decode_scheduler          = false;
};

struct StructuredOutputAdmissionResult {
    bool        allowed = true;
    ErrorCode   code    = ErrorCode::NONE_ERROR;
    std::string message;
};

// True when engine startup config indicates a constraint backend is configured.
// Does not probe runtime backend health (see logits-processor FailFast for that).
bool isStructuredOutputBackendConfigured(const StructuredOutputConfig& engine_cfg) noexcept;

StructuredOutputAdmissionResult checkStructuredOutputAdmission(const StructuredOutputAdmissionContext& ctx);

StructuredOutputAdmissionResult
checkStructuredOutputAdmissionForConfig(const std::shared_ptr<GenerateConfig>& config,
                                        const StructuredOutputConfig&          engine_cfg,
                                        bool                                   batch_decode_scheduler);

}  // namespace rtp_llm
