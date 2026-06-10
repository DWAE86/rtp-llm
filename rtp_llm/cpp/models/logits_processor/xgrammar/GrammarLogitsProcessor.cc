#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <limits>

#include <chrono>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarCompiler.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBackendCpp.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"

namespace rtp_llm {

namespace {

DLTensor makeSingleRowBitmaskView(int32_t* data, int32_t words, int64_t shape_out[2]) {
    DLTensor dl;
    dl.data        = data;
    dl.device      = DLDevice{kDLCPU, 0};
    dl.ndim        = 2;
    dl.dtype       = DLDataType{kDLInt, 32, 1};
    shape_out[0]   = 1;
    shape_out[1]   = words;
    dl.shape       = shape_out;
    dl.strides     = nullptr;
    dl.byte_offset = 0;
    return dl;
}

bool bitmaskAllowsToken(const int32_t* bitmask, int32_t token_id) {
    const int32_t word = bitmask[token_id / 32];
    return (static_cast<uint32_t>(word) & (1u << (token_id % 32))) != 0u;
}

void clearTokenFromBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    bitmask[token_id / 32] &= ~(1u << (token_id % 32));
}

void forceTokenInBitmask(int32_t* bitmask, size_t words, int64_t token_id) {
    std::fill_n(bitmask, words, 0);
    if (token_id < 0 || static_cast<size_t>(token_id / 32) >= words) {
        return;
    }
    bitmask[token_id / 32] |= (1u << (token_id % 32));
}

void clearBitmaskTokenRange(int32_t* bitmask, size_t words, int64_t begin_token, int64_t end_token) {
    if (begin_token < 0 || end_token <= begin_token) {
        return;
    }
    for (int64_t token_id = begin_token; token_id < end_token; ++token_id) {
        clearTokenFromBitmask(bitmask, words, token_id);
    }
}

// Decide the matcher's initial reasoning state from the prompt token stream.
// Returns true ⇒ matcher should start in passthrough (still inside <think>);
// false ⇒ start active (no open think body in prompt). Scans backwards for the
// last think_start / think_end tokens; an unmatched think_start (no matching
// think_end after it) means the prompt ends inside a think body. When only
// think_end_id is known, presence of think_end anywhere in the prompt means
// the body has already closed (active); absence means assume open.
//
// This decouples matcher gating from per-request server flags. The Qwen3 chat
// template auto-injects "<think>\n" regardless of the server's --think_mode,
// so trusting the per-request flag mismatches the actual prompt content and
// makes the matcher either passthrough-forever (Scenario B) or actively mask
// the model's free-text reasoning into JSON-shaped tokens (Scenario A).
bool promptInsideThinkBody(const std::vector<int>&     prompt_tokens,
                           std::optional<int32_t>      think_start_id,
                           std::optional<int32_t>      think_end_id) {
    if (!think_end_id.has_value()) {
        return false;
    }
    const int32_t end_id   = *think_end_id;
    int32_t       start_id = think_start_id.value_or(-1);
    int           last_end_pos   = -1;
    int           last_start_pos = -1;
    for (int i = static_cast<int>(prompt_tokens.size()) - 1; i >= 0; --i) {
        const int32_t tok = static_cast<int32_t>(prompt_tokens[i]);
        if (tok == end_id && last_end_pos < 0) {
            last_end_pos = i;
        }
        if (start_id >= 0 && tok == start_id && last_start_pos < 0) {
            last_start_pos = i;
        }
        if (last_end_pos >= 0 && (start_id < 0 || last_start_pos >= 0)) {
            break;
        }
    }
    if (start_id < 0) {
        // No reliable think_start id: assume "no think_end seen ⇒ inside body"
        // (handles Qwen3 template auto-inserting <think> opener).
        return last_end_pos < 0;
    }
    if (last_start_pos < 0) {
        return false;  // never opened a think body
    }
    // Open if the last think_start is later than the last think_end.
    return last_end_pos < last_start_pos;
}

std::string grammarKindTag(const GenerateStreamPtr& stream) {
    const auto& cfg = stream->generateConfig();
    if (cfg->json_schema.has_value())    return "json_schema";
    if (cfg->regex.has_value())          return "regex";
    if (cfg->ebnf.has_value())           return "ebnf";
    if (cfg->structural_tag.has_value()) return "structural_tag";
    return "none";
}

}  // namespace

GrammarLogitsProcessor::GrammarLogitsProcessor(std::shared_ptr<RtpGrammarMatcher> matcher,
                                               GenerateStreamPtr                  stream):
    resolved_(true), matcher_(std::move(matcher)), stream_(stream) {}

GrammarLogitsProcessor::~GrammarLogitsProcessor() = default;

namespace {
GrammarKeyCpp extractGrammarKey(const GenerateConfig& config) {
    if (config.json_schema.has_value()) {
        return {"json", config.json_schema.value()};
    } else if (config.regex.has_value()) {
        return {"regex", config.regex.value()};
    } else if (config.ebnf.has_value()) {
        return {"ebnf", config.ebnf.value()};
    } else if (config.structural_tag.has_value()) {
        return {"structural_tag", config.structural_tag.value()};
    }
    return {};
}
}  // namespace

BaseLogitsProcessorPtr GrammarLogitsProcessor::tryCreatePending(const std::shared_ptr<GenerateInput>& input) {
    if (!input || !input->generate_config) {
        return nullptr;
    }
    // Non-const: hasNumBeams() (via maxNumBeams) is a non-const accessor.
    auto& config = *input->generate_config;
    if (!config.hasStructuredOutputRequest()) {
        return nullptr;
    }

    auto  prep     = std::shared_ptr<GrammarLogitsProcessor>(new GrammarLogitsProcessor());
    auto& compiler = GrammarCompiler::instance();

    if (!compiler.enabled()) {
        prep->kind_      = Kind::FailFast;
        prep->fail_code_ = ErrorCode::INVALID_PARAMS;
        prep->fail_msg_  =
            "structured output requested but constraint backend is disabled "
            "(check engine startup logs: backend selector is none, tokenizer info is empty, "
            "or backend initialization failed).";
        return prep;
    }

    // Defense-in-depth: beam search + grammar is unsupported (one matcher per
    // stream cannot track divergent per-beam token suffixes; the matcher would
    // not advance and the model would emit schema-illegal tokens). The Python
    // generate_config.validate() already rejects this on the gRPC path, but
    // native C++ HTTP / non-Python gRPC callers bypass that — reject here too.
    if (config.hasNumBeams()) {
        prep->kind_      = Kind::FailFast;
        prep->fail_code_ = ErrorCode::INVALID_PARAMS;
        prep->fail_msg_  =
            "structured output is not supported together with beam search "
            "(num_beams > 1). Submit the request with num_beams=1 or without the constraint.";
        return prep;
    }
    if (config.num_return_sequences > 1) {
        prep->kind_      = Kind::FailFast;
        prep->fail_code_ = ErrorCode::INVALID_PARAMS;
        prep->fail_msg_  = "structured output is not supported with multiple return sequences";
        return prep;
    }

    prep->kind_              = Kind::Compile;
    prep->key_               = extractGrammarKey(config);
    prep->require_reasoning_ = config.in_think_mode;
    prep->future_            = compiler.submit(prep->key_);
    prep->deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(compiler.compileTimeoutMs());
    RTP_LLM_LOG_DEBUG("grammar processor created (pending): key=%s, require_reasoning=%d",
                      prep->key_.brief().c_str(), static_cast<int>(prep->require_reasoning_));
    return prep;
}

bool GrammarLogitsProcessor::needsPreparation() const {
    // True until prepare() reaches a terminal result. Covers both the Ready path
    // (matcher installed) and the FailFast/compile-error path (matcher stays null
    // but the processor is resolved), so a re-query never re-arms the gate.
    return !resolved_;
}

void GrammarLogitsProcessor::installMatcher(GenerateStream& stream, const GrammarReadyPayload& payload) {
    auto& compiler = GrammarCompiler::instance();

    // Arm the reasoning gate whenever the engine has a resolved think_end_id,
    // independent of the per-request server flag. Per-request flags
    // (in_think_mode) and the actual prompt-side <think> content are routinely
    // out of sync — the Qwen3 chat template injects "<think>\n" regardless of
    // --think_mode, so the matcher must look at the prompt itself to decide
    // initial passthrough vs active.
    const bool reasoner_capable = compiler.hasReasoner();
    matcher_                                 = compiler.createMatcher(payload.compiled, reasoner_capable);
    matcher_->mutableStats().cache_hit       = payload.cache_hit;
    matcher_->mutableStats().compile_time_us = payload.compile_time_us;
    matcher_->mutableStats().dispatch_type   = key_.key_type;

    // Decide initial reasoning state by scanning the actual prompt:
    //   * prompt ends inside a still-open <think> body  ⇒ start passthrough
    //   * prompt has no open <think> (or already closed) ⇒ start active
    //   * matcher not reasoner-capable                   ⇒ initReasoning is a no-op
    bool in_think_body = false;
    if (reasoner_capable) {
        const int input_len = stream.inputLength();
        if (input_len > 0) {
            auto prompt_tokens = stream.completeTokenIdsVec(0);
            if (static_cast<int>(prompt_tokens.size()) > input_len) {
                prompt_tokens.resize(input_len);
            }
            in_think_body = promptInsideThinkBody(
                prompt_tokens, matcher_->thinkStartId(), matcher_->thinkEndId());
        }
    }
    matcher_->initReasoning(in_think_body);
    stream_ = stream.sharedThis();

    // Snapshot any tokens already present on the stream BEFORE replaying them into
    // the fresh matcher. A normal prefill stream is prepared while WAITING (no
    // output tokens yet), so this is a no-op; the PD-decode side, however, arrives
    // with prefill bonus token(s) that must be replayed to sync matcher state.
    // Reads are lock-safe under the stream's mutex_ (prepare holds it).
    std::vector<int32_t> prefill_tokens;
    const size_t         output_len = stream.outputTokenLen();
    if (output_len > 0) {
        auto      all_tokens = stream.completeTokenIdsVec(0);
        const int input_len  = stream.inputLength();
        prefill_tokens.reserve(output_len);
        for (size_t i = 0; i < output_len; ++i) {
            prefill_tokens.push_back(static_cast<int32_t>(all_tokens[input_len + i]));
        }
    }
    if (!prefill_tokens.empty()) {
        (void)advanceMatcher(prefill_tokens, MatcherAdvance::ReplayOnly, /*caller_holds_stream_lock=*/true);
    } else {
        matcher_synced_ = true;
    }
    accepted_token_len_ = matcher_ ? matcher_->numAcceptedTokens() : 0;
    eos_token_id_       = stream.specialTokens().eos_token_id;
    grammar_state_.replayed_token_count = static_cast<int32_t>(output_len);
    if (matcher_) {
        auto snap              = matcher_->snapshot();
        grammar_state_.num_accepted           = snap.num_accepted;
        grammar_state_.tokens_after_think_end = snap.tokens_after_think_end;
        grammar_state_.finished               = snap.finished;
        grammar_state_.stats                  = snap.stats;
    }
}

void GrammarLogitsProcessor::logGrammarLifecycle(const char* phase,
                                                 const char* source,
                                                 int64_t     stream_id) const {
    if (stream_id < 0) {
        if (auto stream = stream_.lock()) {
            stream_id = stream->streamId();
        }
    }
    const int64_t num_accepted     = matcher_ ? matcher_->numAcceptedTokens() : -1;
    const int64_t mask_apply_count = matcher_ ? static_cast<int64_t>(matcher_->stats().mask_apply_count) : -1;
    const int     terminated       = matcher_ && matcher_->isTerminated() ? 1 : 0;
    const int     finished         = matcher_ && matcher_->finished() ? 1 : 0;
    RTP_LLM_LOG_DEBUG("[grammar_lifecycle] stream=%ld phase=%s source=%s num_accepted=%ld "
                     "mask_apply_count=%ld terminated=%d finished=%d synced=%d",
                     stream_id,
                     phase,
                     source,
                     static_cast<long>(num_accepted),
                     static_cast<long>(mask_apply_count),
                     terminated,
                     finished,
                     static_cast<int>(matcher_synced_));
}

bool GrammarLogitsProcessor::ensureMatcherSynced() {
    if (!matcher_) {
        return false;
    }
    if (matcher_synced_) {
        return true;
    }
    // If matcher is not synced at this point, the commit path (updateStatus)
    // has not yet delivered the pending tokens. Reading stream token data here
    // without holding stream->mutex_ would be a data race when called from the
    // async spec-verify thread. Instead of attempting the read, report that the
    // matcher is not ready — callers (tryAcceptAndFillBitmask) will degrade
    // gracefully (return propose_step=0, disabling grammar verify for this tick).
    RTP_LLM_LOG_DEBUG("[grammar] ensureMatcherSynced: matcher not synced, deferring to commit path");
    return false;
}

bool GrammarLogitsProcessor::advanceMatcher(const std::vector<int32_t>& tokens,
                                             MatcherAdvance              mode,
                                             bool                        caller_holds_stream_lock) {
    if (tokens.empty()) {
        matcher_synced_ = true;
        return true;
    }
    if (!matcher_) {
        RTP_LLM_LOG_WARNING("[grammar] advanceMatcher: matcher not installed");
        return false;
    }

    const char* source = mode == MatcherAdvance::Commit ? "commit" : "replay";
    logGrammarLifecycle(source, source);

    for (int32_t tok : tokens) {
        if (matcher_->isTerminated() || matcher_->finished()) {
            break;
        }
        if (!matcher_->acceptToken(tok)) {
            reported_error_.store(true, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("[grammar] advanceMatcher rejected token %d source=%s num_accepted=%ld passthrough=%d",
                                tok,
                                source,
                                matcher_->numAcceptedTokens(),
                                static_cast<int>(matcher_->isPassthroughForMask()));
            if (auto stream = stream_.lock()) {
                if (mode == MatcherAdvance::Commit || caller_holds_stream_lock) {
                    stream->reportErrorWithoutLock(
                        ErrorCode::INVALID_PARAMS,
                        std::string("grammar ") + source + " error: parser rejected token "
                            + std::to_string(tok));
                } else {
                    stream->reportError(
                        ErrorCode::INVALID_PARAMS,
                        std::string("grammar ") + source + " error: parser rejected token "
                            + std::to_string(tok));
                }
            } else {
                matcher_->markFinished();
            }
            matcher_synced_ = true;
            return false;
        }
    }

    matcher_synced_ = true;
    if (mode == MatcherAdvance::Commit) {
        checkTermination();
    }
    return true;
}

PrepareState GrammarLogitsProcessor::prepare(GenerateStream& stream) {
    // Idempotent once resolved: re-polls after a Ready/Failed result are
    // side-effect-free (no repeated error reporting). Pending re-polls (compile
    // in flight / PD-decode token wait) keep running the body below.
    if (resolved_) {
        return matcher_ ? PrepareState::Ready : PrepareState::Failed;
    }

    const PrepareState state = [&]() -> PrepareState {
        if (kind_ == Kind::FailFast) {
            stream.reportErrorWithoutLock(fail_code_, fail_msg_);
            return PrepareState::Failed;
        }

        if (!future_.valid()) {
            stream.reportErrorWithoutLock(ErrorCode::EXECUTION_EXCEPTION, "grammar preparation future invalid");
            return PrepareState::Failed;
        }

        if (future_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            if (std::chrono::steady_clock::now() >= deadline_) {
                RTP_LLM_LOG_WARNING("stream [%ld] grammar wait timeout: limit_ms=%lld",
                                    stream.streamId(),
                                    static_cast<long long>(GrammarCompiler::instance().compileTimeoutMs()));
                stream.reportErrorWithoutLock(ErrorCode::GENERATE_TIMEOUT, "Grammar preprocessing timed out");
                return PrepareState::Failed;
            }
            return PrepareState::Pending;
        }

        GrammarReadyPayload payload;
        try {
            payload = future_.get();
        } catch (const std::exception& e) {
            stream.reportErrorWithoutLock(ErrorCode::INVALID_PARAMS,
                                          std::string("grammar compile error: ") + e.what());
            return PrepareState::Failed;
        }

        // PD-decode: allocate/prepare normally runs before localGenerate ingest;
        // installMatcher replays any output already on the stream via syncMatcherTokens.

        if (payload.is_invalid || !payload.compiled) {
            std::string err = payload.error_msg.empty() ? "unknown compile error" : payload.error_msg;
            stream.reportErrorWithoutLock(ErrorCode::INVALID_PARAMS,
                                          "Failed to compile " + key_.key_type + " grammar: " + err);
            return PrepareState::Failed;
        }

        try {
            installMatcher(stream, payload);
            logGrammarLifecycle("compile", "prepare", stream.streamId());
        } catch (const std::exception& e) {
            RTP_LLM_LOG_ERROR("stream [%ld] grammar install failed: %s", stream.streamId(), e.what());
            stream.reportErrorWithoutLock(ErrorCode::EXECUTION_EXCEPTION,
                                          std::string("grammar matcher install failed: ") + e.what());
            return PrepareState::Failed;
        }
        RTP_LLM_LOG_DEBUG("stream [%ld] grammar ready: key=%s", stream.streamId(), key_.brief().c_str());
        return PrepareState::Ready;
    }();

    // Latch the terminal result so subsequent polls are no-ops and
    // needsPreparation() reports false.
    if (state != PrepareState::Pending) {
        resolved_ = true;
    }
    return state;
}

void GrammarLogitsProcessor::prepareNormalAsyncUpdate(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (num_new_tokens <= 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    pending_async_token_len_ = std::max(pending_async_token_len_, accepted_token_len_ + num_new_tokens);
    if (new_tokens.defined() && new_tokens.is_cuda()) {
        last_mask_device_ = new_tokens.device();
    }
}

int64_t GrammarLogitsProcessor::acceptedTokenLen() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return accepted_token_len_;
}

void GrammarLogitsProcessor::syncAcceptedTokenLenLocked() {
    accepted_token_len_ = matcher_ ? matcher_->numAcceptedTokens() : 0;
}

void GrammarLogitsProcessor::rebuildDeviceMaskStateLocked() {
    device_mask_state_ = buildDeviceMaskStateLocked(last_mask_device_.value_or(c10::Device(c10::DeviceType::CPU)));
}

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::getDeviceMaskState(const c10::Device& device) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    last_mask_device_ = device;
    if (pending_async_token_len_ > accepted_token_len_) {
        static constexpr auto kMaskWaitTimeout = std::chrono::milliseconds(5000);
        bool resolved = state_cv_.wait_for(lock, kMaskWaitTimeout, [this]() {
            return pending_async_token_len_ <= accepted_token_len_ || reported_error_.load() || !matcher_
                   || matcher_->finished();
        });
        if (!resolved) {
            RTP_LLM_LOG_WARNING("[grammar] getDeviceMaskState timed out waiting for token sync "
                                "(pending=%ld accepted=%ld), treating as finished",
                                pending_async_token_len_, accepted_token_len_);
            DeviceMaskState state;
            state.mode      = DeviceMaskMode::FINISHED;
            state.token_len = accepted_token_len_;
            state.device    = device;
            return state;
        }
    }

    if (device_mask_state_.mode != DeviceMaskMode::UNSET && device_mask_state_.token_len == accepted_token_len_
        && device_mask_state_.device == device) {
        return device_mask_state_;
    }

    device_mask_state_ = buildDeviceMaskStateLocked(device);
    return device_mask_state_;
}

GrammarLogitsProcessor::DeviceMaskState GrammarLogitsProcessor::buildDeviceMaskStateLocked(const c10::Device& device) {
    DeviceMaskState state;
    state.token_len = accepted_token_len_;
    state.device    = device;

    if (!matcher_ || matcher_->finished()) {
        state.mode = DeviceMaskMode::FINISHED;
        return state;
    }
    if (matcher_->isTerminated()) {
        state.mode = DeviceMaskMode::TERMINATED;
        return state;
    }
    if (matcher_->isPassthroughForMask()) {
        state.mode = DeviceMaskMode::PASSTHROUGH;
        return state;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }

    const int32_t words   = (grammar_vocab_size + 31) / 32;
    auto          bitmask = at::full({1, words}, -1, at::dtype(at::kInt));
    int64_t       dl_shape[2];
    DLTensor      dl      = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words, dl_shape);
    if (!matcher_->fillBitmask(&dl, 0)) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }

    // TODO(grammar-perf): reuse a persistent pinned/device buffer instead of
    // allocating a fresh V-element bool tensor every token. Consider applying
    // the int32 bitmask directly on GPU via a kernel (bitmask_ops Triton).
    auto mask_options = torch::TensorOptions().dtype(torch::kBool);
    if (device.is_cuda()) {
        mask_options = mask_options.pinned_memory(true);
    }
    auto           vocab_mask  = torch::empty({grammar_vocab_size}, mask_options);
    bool*          mask_ptr    = vocab_mask.data_ptr<bool>();
    const int32_t* bitmask_ptr = bitmask.data_ptr<int32_t>();
    for (int32_t token_id = 0; token_id < grammar_vocab_size; ++token_id) {
        mask_ptr[token_id] = !bitmaskAllowsToken(bitmask_ptr, token_id);
    }

    state.mode = DeviceMaskMode::MASK;
    publishMaskToDevice(state, vocab_mask, device);
    return state;
}

void GrammarLogitsProcessor::publishMaskToDevice(DeviceMaskState&   state,
                                                 torch::Tensor      vocab_mask,
                                                 const c10::Device& device) {
    if (!device.is_cuda()) {
        state.vocab_mask = vocab_mask;
        return;
    }

    state.vocab_mask = vocab_mask.to(device, /*non_blocking=*/true);
#if USING_CUDA || USING_ROCM
    state.ready_event = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    state.ready_event->record(cuda_graph::graphGetCurrentStream());
#endif
}

void GrammarLogitsProcessor::applyDeviceMaskState(const torch::Tensor& logits, const DeviceMaskState& state) {
    switch (state.mode) {
        case DeviceMaskMode::FINISHED:
        case DeviceMaskMode::NOOP:
        case DeviceMaskMode::UNSET:
            return;
        case DeviceMaskMode::TERMINATED:
            forceToken(logits, eos_token_id_);
            return;
        case DeviceMaskMode::PASSTHROUGH:
            maskToken(logits, eos_token_id_);
            return;
        case DeviceMaskMode::MASK:
            break;
    }

    if (!state.vocab_mask.defined()) {
        return;
    }
#if USING_CUDA || USING_ROCM
    if (state.ready_event && logits.is_cuda()) {
        state.ready_event->block(cuda_graph::graphGetCurrentStream());
    }
#endif
    auto mask = state.vocab_mask;
    if (mask.device() != logits.device()) {
        mask = mask.to(logits.device(), /*non_blocking=*/true);
    }
    const int64_t mask_vocab_size = std::min<int64_t>(logits.size(0), mask.size(0));
    if (mask_vocab_size > 0) {
        logits.narrow(0, 0, mask_vocab_size)
            .masked_fill_(mask.narrow(0, 0, mask_vocab_size), BaseLogitsProcessor::neg_inf);
    }
    if (mask.size(0) < logits.size(0)) {
        logits.narrow(0, mask.size(0), logits.size(0) - mask.size(0)).fill_(BaseLogitsProcessor::neg_inf);
    }
}

void GrammarLogitsProcessor::forceToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits.fill_(BaseLogitsProcessor::neg_inf);
    logits[token_id] = 1;
}

void GrammarLogitsProcessor::maskToken(const torch::Tensor& logits, int64_t token_id) {
    if (token_id < 0 || token_id >= logits.size(0)) {
        return;
    }
    logits[token_id] = BaseLogitsProcessor::neg_inf;
}

void GrammarLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (!matcher_) {
        return;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size == 0) {
        return;
    }
    if (batch_size != 1) {
        RTP_LLM_LOG_WARNING("grammar logits processor only supports single sequence decoding");
        return;
    }
    if (inputs.finished_mask.defined()) {
        const auto* finished = inputs.finished_mask.data_ptr<uint8_t>();
        if (finished[start_idx]) {
            return;
        }
    }

    auto logits = inputs.logits.narrow(0, start_idx, 1);
    auto state  = getDeviceMaskState(logits.device());
    applyDeviceMaskState(logits[0], state);
}

void GrammarLogitsProcessor::processSpeculative(const SamplerInputs&        inputs,
                                                size_t                      start_idx,
                                                size_t                      finish_idx,
                                                const std::vector<int32_t>& draft_prefix) {
    if (draft_prefix.empty()) {
        process(inputs, start_idx, finish_idx);
        return;
    }
    // Ineligible processor on the MTP verify path (e.g. reported_error_=true).
    // Apply current mask state as a conservative fallback for the first position
    // so this token is at least grammar-constrained, even if not multi-step.
    process(inputs, start_idx, finish_idx);
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (!matcher_ || matcher_->isTerminated() || matcher_->finished()) {
        return;
    }

    RTP_LLM_CHECK(new_tokens.dim() == 2);
    RTP_LLM_CHECK(new_tokens.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(new_tokens.size(1) >= num_new_tokens);
    RTP_LLM_CHECK(new_tokens.is_contiguous());

    const int              batch_size = static_cast<int>(new_tokens.size(0));
    const int              stride     = static_cast<int>(new_tokens.size(1));
    const auto*            data       = new_tokens.data_ptr<int32_t>();
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(batch_size * num_new_tokens));
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_new_tokens; ++j) {
            tokens.push_back(data[i * stride + j]);
        }
    }

    RTP_LLM_PROFILE_SCOPE("grammar.acceptToken");
    const int64_t trace_stream_id = [&]() {
        if (auto s = stream_.lock()) {
            return s->streamId();
        }
        return static_cast<int64_t>(-1);
    }();
    logGrammarLifecycle("commit", "notifyCommit", trace_stream_id);
    RTP_LLM_LOG_DEBUG("[grammar_commit_trace] commit start: stream=%ld num_accepted=%ld mask_apply_count=%ld "
                      "num_new_tokens=%d batch=%d passthrough=%d",
                      trace_stream_id,
                      matcher_->numAcceptedTokens(),
                      static_cast<long>(matcher_->stats().mask_apply_count),
                      num_new_tokens,
                      batch_size,
                      static_cast<int>(matcher_->isPassthroughForMask()));

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!advanceMatcher(tokens, MatcherAdvance::Commit)) {
            state_cv_.notify_all();
            grammar_state_.replayed_token_count += num_new_tokens;
            return;
        }
        syncAcceptedTokenLenLocked();
        pending_async_token_len_ = accepted_token_len_;
        rebuildDeviceMaskStateLocked();
        state_cv_.notify_all();
    }
    grammar_state_.replayed_token_count += num_new_tokens;
    if (matcher_) {
        auto snap              = matcher_->snapshot();
        grammar_state_.num_accepted           = snap.num_accepted;
        grammar_state_.tokens_after_think_end = snap.tokens_after_think_end;
        grammar_state_.finished               = snap.finished;
        grammar_state_.stats                  = snap.stats;
    }

    if (auto stream = stream_.lock()) {
        RTP_LLM_LOG_DEBUG("[sp_accept_trace] stream_id=%ld grammar=%s propose_step=%zu accept_len=%d",
                          stream->streamId(),
                          grammarKindTag(stream).c_str(),
                          stream->getProposeStep(),
                          num_new_tokens);
    }
}

void GrammarLogitsProcessor::checkTermination() {
    if (matcher_->isTerminated()) {
        matcher_->markFinished();
        logGrammarLifecycle("done", "commit_terminate");
        if (auto stream = stream_.lock()) {
            if (stream->isActive()) {
                // checkTermination is invoked from updateStatus on the commit
                // path only — the sole grammar-driven GenerateDone source.
                // Call sites already hold the stream's mutex_ (see
                // GenerateStream::update / specUpdate / moveToNext).
                // reportEvent would re-acquire the non-recursive mutex and
                // deadlock; use reportEventWithoutLock instead.
                stream->reportEventWithoutLock(StreamEvents::GenerateDone);
            }
        }
    } else {
        if (auto stream = stream_.lock()) {
            if (!stream->isActive()) {
                matcher_->markFinished();
            }
        }
    }
}

void GrammarLogitsProcessor::replayCommittedTokens(const std::vector<int32_t>& tokens) {
    auto stream = stream_.lock();
    if (!matcher_ || !stream) {
        return;
    }

    const size_t output_len = stream->outputTokenLen();

    if (grammar_state_.pd_bonus_ingested && output_len <= static_cast<size_t>(grammar_state_.replayed_token_count)) {
        RTP_LLM_LOG_INFO("[grammar_pd_replay] idempotent skip: stream=%ld replayed_token_count=%d",
                         stream->streamId(),
                         grammar_state_.replayed_token_count);
        return;
    }

    std::vector<int32_t> delta;
    if (output_len > static_cast<size_t>(grammar_state_.replayed_token_count)) {
        auto                 all_tokens = stream->completeTokenIdsVec(0);
        const int            input_len  = stream->inputLength();
        const size_t         start      = static_cast<size_t>(grammar_state_.replayed_token_count);
        delta.reserve(output_len - start);
        for (size_t i = start; i < output_len; ++i) {
            delta.push_back(static_cast<int32_t>(all_tokens[input_len + i]));
        }
    } else if (!tokens.empty()) {
        delta = tokens;
    }

    if (delta.empty()) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        matcher_synced_ = false;
        if (!advanceMatcher(delta, MatcherAdvance::ReplayOnly, /*caller_holds_stream_lock=*/true)) {
            RTP_LLM_LOG_WARNING("[grammar_pd_replay] replay failed stream=%ld",
                                stream->streamId());
            return;
        }
        grammar_state_.replayed_token_count =
            static_cast<int32_t>(std::max(output_len, static_cast<size_t>(grammar_state_.replayed_token_count + delta.size())));
        if (!tokens.empty() && tokens.size() == 1) {
            grammar_state_.pd_bonus_ingested = 1;
        }
        if (matcher_) {
            auto snap              = matcher_->snapshot();
            grammar_state_.num_accepted           = snap.num_accepted;
            grammar_state_.tokens_after_think_end = snap.tokens_after_think_end;
            grammar_state_.finished               = snap.finished;
            grammar_state_.stats                  = snap.stats;
        }
        syncAcceptedTokenLenLocked();
        pending_async_token_len_ = accepted_token_len_;
        rebuildDeviceMaskStateLocked();
        state_cv_.notify_all();
    }

    RTP_LLM_LOG_INFO("[grammar_pd_replay] replayed %zu tokens stream=%ld replayed_total=%d",
                     delta.size(),
                     stream->streamId(),
                     grammar_state_.replayed_token_count);
}

void GrammarLogitsProcessor::finalizeAfterReplayIfComplete() {
    if (!matcher_ || matcher_->finished()) {
        return;
    }
    if (matcher_->isTerminated()) {
        checkTermination();
    }
}

int GrammarLogitsProcessor::previewCommitTokenCount(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) {
        return 0;
    }
    if (!matcher_) {
        return static_cast<int>(tokens.size());
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!matcher_synced_) {
        if (!ensureMatcherSynced()) {
            RTP_LLM_LOG_WARNING("[grammar] previewCommitTokenCount: matcher not synced; keeping full accept_len");
            return static_cast<int>(tokens.size());
        }
    }
    if (matcher_->isTerminated() || matcher_->finished()) {
        return 0;
    }

    int  accepted             = 0;
    bool terminated_mid_preview = false;
    try {
        for (int32_t tok : tokens) {
            if (matcher_->isTerminated() || matcher_->finished()) {
                break;
            }
            if (!matcher_->acceptToken(tok)) {
                break;
            }
            ++accepted;
            if (matcher_->isTerminated()) {
                terminated_mid_preview = true;
                break;
            }
        }
        if (accepted > 0) {
            matcher_->rollback(accepted);
        }
        if (terminated_mid_preview && matcher_->isTerminated()) {
            // Rollback failed to unset terminated state — exclude the
            // terminal token from the accepted count to avoid permanently
            // disabling spec acceptance.
            RTP_LLM_LOG_WARNING("[grammar] previewCommitTokenCount: rollback did not clear terminated; "
                                "capping accepted from %d to %d", accepted, accepted - 1);
            accepted = std::max(0, accepted - 1);
        }
        RTP_LLM_LOG_DEBUG("[grammar_commit_trace] previewCommitTokenCount proposed=%zu accepted=%d",
                          tokens.size(),
                          accepted);
    } catch (...) {
        if (accepted > 0) {
            try {
                matcher_->rollback(accepted);
            } catch (...) {
                matcher_->markFinished();
            }
        }
        throw;
    }
    return accepted;
}

void GrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& /* src_batch_indices */) {}

bool GrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr && !reported_error_.load(std::memory_order_relaxed);
}

int GrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }
    RTP_LLM_CHECK_WITH_INFO(
        request.bitmask_size_int32 >= static_cast<size_t>((request.vocab_size + 31) / 32),
        "bitmask buffer too small for vocab_size");
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (!matcher_synced_ && !ensureMatcherSynced()) {
        return 0;
    }

    const int  P               = request.propose_step;
    const auto W               = request.bitmask_size_int32;
    int        accepted_prefix = 0;
    int        cap             = P;

    auto fill_row = [&](int32_t* row) {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        if (matcher_->finished()) {
            return;
        }
        if (matcher_->isTerminated()) {
            forceTokenInBitmask(row, W, eos_token_id_);
            return;
        }
        if (matcher_->isPassthroughForMask()) {
            clearTokenFromBitmask(row, W, eos_token_id_);
            return;
        }

        const int32_t grammar_vocab_size = matcher_->vocabSize();
        const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);
        RTP_LLM_CHECK_WITH_INFO(
            grammar_words <= W, "grammar vocab bitmask exceeds model vocab bitmask in MTP verify");

        int64_t  dl_shape[2];
        DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
        if (!matcher_->fillBitmask(&dl, 0) && matcher_->isPassthroughForMask()) {
            std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
            clearTokenFromBitmask(row, W, eos_token_id_);
            return;
        }
        clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(request.vocab_size));
    };

    try {
        for (int offset = 0; offset <= P; ++offset) {
            int32_t* row = request.bitmask_cpu_out + offset * W;
            fill_row(row);
            if (offset == P) {
                break;
            }

            const int32_t draft_token = request.draft_tokens[offset];
            if (draft_token < 0 || static_cast<size_t>(draft_token) >= request.vocab_size
                || !bitmaskAllowsToken(row, draft_token)) {
                cap = offset;
                break;
            }
            if (!matcher_->acceptToken(draft_token)) {
                cap = offset;
                break;
            }
            ++accepted_prefix;
        }

        if (accepted_prefix > 0) {
            matcher_->rollback(accepted_prefix);
        }
    } catch (...) {
        if (accepted_prefix > 0) {
            try { matcher_->rollback(accepted_prefix); } catch (...) { matcher_->markFinished(); }
        }
        throw;
    }
    return cap;
}

}  // namespace rtp_llm
