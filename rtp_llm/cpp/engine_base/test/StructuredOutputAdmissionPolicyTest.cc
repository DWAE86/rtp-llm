#include "rtp_llm/cpp/engine_base/StructuredOutputAdmissionPolicy.h"

#include "gtest/gtest.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"

namespace rtp_llm {

namespace {

GenerateConfig grammarLikeConfig() {
    GenerateConfig cfg;
    cfg.json_schema = R"({"type":"object"})";
    return cfg;
}

StructuredOutputConfig enabledEngineConfig() {
    StructuredOutputConfig cfg;
    cfg.grammar_backend     = "xgrammar";
    cfg.tokenizer_info_json = "{}";
    return cfg;
}

}  // namespace

TEST(StructuredOutputAdmissionPolicyTest, AllowsPlainRequest) {
    GenerateConfig cfg;
    auto           result = checkStructuredOutputAdmissionForConfig(std::make_shared<GenerateConfig>(cfg),
                                                          enabledEngineConfig(),
                                                          false);
    EXPECT_TRUE(result.allowed);
}

TEST(StructuredOutputAdmissionPolicyTest, RejectsBeamWithStructuredOutput) {
    auto cfg      = grammarLikeConfig();
    cfg.num_beams = 2;
    auto result   = checkStructuredOutputAdmissionForConfig(std::make_shared<GenerateConfig>(cfg),
                                                          enabledEngineConfig(),
                                                          false);
    EXPECT_FALSE(result.allowed);
}

TEST(StructuredOutputAdmissionPolicyTest, RejectsBatchDecodeScheduler) {
    auto result = checkStructuredOutputAdmissionForConfig(std::make_shared<GenerateConfig>(grammarLikeConfig()),
                                                          enabledEngineConfig(),
                                                          true);
    EXPECT_FALSE(result.allowed);
}

TEST(StructuredOutputAdmissionPolicyTest, RejectsWhenBackendNotConfigured) {
    StructuredOutputConfig disabled;
    disabled.grammar_backend = "none";
    auto result              = checkStructuredOutputAdmissionForConfig(std::make_shared<GenerateConfig>(grammarLikeConfig()),
                                                          disabled,
                                                          false);
    EXPECT_FALSE(result.allowed);
}

}  // namespace rtp_llm
