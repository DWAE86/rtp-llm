#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarLogitsProcessor.h"

#include <algorithm>
#include <cstring>
#include <list>
#include <limits>

#include <chrono>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/GrammarCompiler.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/RtpGrammarMatcher.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBackend.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#if USING_CUDA || USING_ROCM
#include "rtp_llm/cpp/models/logits_processor/xgrammar/xgrammar_kernels.h"
#endif

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

enum class VerifyRowState {
    Active,
    Finished,
    Terminated,
};

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

    // Reasoning admission: an in-think request must carry end_think_token_ids
    // — without them the gate has nothing to scan for and the parser would stay
    // frozen forever. Capture both flags so installMatcher can configure the
    // matcher's KMP scanner; non-think requests leave require_reasoning_=false
    // and behave as plain grammar.
    if (config.in_think_mode) {
        if (config.end_think_token_ids.empty()) {
            prep->kind_      = Kind::FailFast;
            prep->fail_code_ = ErrorCode::INVALID_PARAMS;
            prep->fail_msg_  = "grammar-constrained thinking requires non-empty end_think_token_ids";
            return prep;
        }
        prep->require_reasoning_   = true;
        prep->think_end_token_ids_ = config.end_think_token_ids;
    }

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

    prep->kind_   = Kind::Compile;
    prep->key_    = extractGrammarKey(config);
    prep->future_ = compiler.submit(prep->key_);
    prep->deadline_ = std::chrono::steady_clock::now() + std::chrono::milliseconds(compiler.compileTimeoutMs());
    RTP_LLM_LOG_DEBUG("grammar processor created (pending): key=%s", prep->key_.brief().c_str());
    return prep;
}

bool GrammarLogitsProcessor::needsPreparation() const {
    // True until prepare() reaches a terminal result. Covers both the Ready path
    // (matcher installed) and the FailFast/compile-error path (matcher stays null
    // but the processor is resolved), so a re-query never re-arms the gate.
    return !resolved_;
}

void GrammarLogitsProcessor::installMatcher(GenerateStream& stream, const GrammarReadyPayload& payload) {
    const bool terminate_without_stop_token = key_.key_type == "json";
    matcher_ = GrammarCompiler::instance().createMatcher(payload.compiled,
                                                         terminate_without_stop_token,
                                                         require_reasoning_,
                                                         think_end_token_ids_);
    stream_  = stream.sharedThis();

    // Reasoning gate: enter passthrough so prefill-replay and the first decode
    // step see the matcher's KMP scanner instead of the active parser. Must
    // run before any acceptToken (the matcher's own contract).
    if (matcher_ && require_reasoning_) {
        matcher_->initReasoning(/*in_think_body=*/true);
    }

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
    }
    accepted_token_len_ = matcher_ ? matcher_->numAcceptedTokens() : 0;
    eos_token_id_       = stream.specialTokens().eos_token_id;
}

void GrammarLogitsProcessor::logGrammarLifecycle(const char* phase,
                                                 const char* source,
                                                 int64_t     stream_id) const {
    if (stream_id < 0) {
        if (auto stream = stream_.lock()) {
            stream_id = stream->streamId();
        }
    }
    const int64_t num_accepted = matcher_ ? matcher_->numAcceptedTokens() : -1;
    const int     terminated   = matcher_ && matcher_->isTerminated() ? 1 : 0;
    const int     finished     = matcher_ && matcher_->finished() ? 1 : 0;
    RTP_LLM_LOG_DEBUG("[grammar_lifecycle] stream=%ld phase=%s source=%s num_accepted=%ld "
                      "terminated=%d finished=%d",
                      stream_id,
                      phase,
                      source,
                      static_cast<long>(num_accepted),
                      terminated,
                      finished);
}

bool GrammarLogitsProcessor::advanceMatcher(const std::vector<int32_t>& tokens,
                                             MatcherAdvance              mode,
                                             bool                        caller_holds_stream_lock) {
    if (tokens.empty()) {
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
            // Always markFinished so subsequent getDeviceMaskState routes through
            // FINISHED early-return rather than fillBitmask on a parser left in
            // an indeterminate post-reject state. Aligns with dsv4 baseline.
            matcher_->markFinished();
            RTP_LLM_LOG_WARNING("[grammar] advanceMatcher rejected token %d source=%s num_accepted=%ld",
                                tok,
                                source,
                                matcher_->numAcceptedTokens());
            const bool stream_lock_held = (mode == MatcherAdvance::Commit) || caller_holds_stream_lock;
            const std::string err_msg =
                std::string("grammar ") + source + " error: parser rejected token " + std::to_string(tok);
            if (error_reporter_) {
                reportErrorViaReporter(ErrorCode::INVALID_PARAMS, err_msg, stream_lock_held);
            }
            return false;
        }
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
        RTP_LLM_LOG_DEBUG("[grammar] getDeviceMaskState waiting for token sync: pending=%ld accepted=%ld",
                          pending_async_token_len_, accepted_token_len_);
        static constexpr auto kMaskWaitTimeout = std::chrono::milliseconds(5000);
        bool resolved = state_cv_.wait_for(lock, kMaskWaitTimeout, [this]() {
            return pending_async_token_len_ <= accepted_token_len_ || reported_error_.load() || !matcher_
                   || matcher_->finished();
        });
        if (!resolved) {
            // Token sync failed within budget. Silently returning FINISHED would let
            // the sampler emit unconstrained tokens with no error signal (user gets
            // schema-illegal output and never knows why). Surface it as a stream
            // error and route through FINISHED so subsequent process() short-circuits.
            RTP_LLM_LOG_WARNING("[grammar] getDeviceMaskState timed out waiting for token sync "
                                "(pending=%ld accepted=%ld); reporting stream error",
                                pending_async_token_len_, accepted_token_len_);
            reported_error_.store(true, std::memory_order_relaxed);
            if (matcher_) {
                matcher_->markFinished();
            }
            // process() runs without holding the stream's mutex_; reporters that
            // need the lock must acquire it themselves.
            reportErrorViaReporter(ErrorCode::GENERATE_TIMEOUT,
                                   "grammar token sync timed out (pending=" + std::to_string(pending_async_token_len_)
                                       + " accepted=" + std::to_string(accepted_token_len_) + ")",
                                   /*stream_lock_held=*/false);
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
    // Reasoning passthrough: still inside the think body, parser is frozen.
    // Emit an allow-all (no-op) mask; updateStatus / spec verify keep stepping
    // the matcher's KMP scanner so the transition into active grammar lands
    // automatically once think_end_token_ids matches.
    if (matcher_->isPassthroughForMask()) {
        state.mode = DeviceMaskMode::PASSTHROUGH;
        return state;
    }

    const int32_t grammar_vocab_size = matcher_->vocabSize();
    if (grammar_vocab_size <= 0) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }

    const int32_t words = (grammar_vocab_size + 31) / 32;
    if (!reusable_bitmask_cpu_.defined() || reusable_mask_words_ < words) {
        reusable_bitmask_cpu_ = at::full({1, words}, -1, at::dtype(at::kInt));
        reusable_mask_words_  = words;
    } else {
        reusable_bitmask_cpu_.fill_(-1);
    }
    auto bitmask = reusable_bitmask_cpu_.narrow(1, 0, words);
    int64_t       dl_shape[2];
    DLTensor      dl      = makeSingleRowBitmaskView(bitmask.data_ptr<int32_t>(), words, dl_shape);
    if (!matcher_->fillBitmask(&dl, 0)) {
        state.mode = DeviceMaskMode::NOOP;
        return state;
    }

    state.grammar_vocab_size = grammar_vocab_size;
#if USING_CUDA || USING_ROCM
    if (device.is_cuda()) {
        if (!reusable_bitmask_gpu_.defined() || reusable_bitmask_gpu_.size(1) < words
            || reusable_bitmask_gpu_.device() != device) {
            reusable_bitmask_gpu_ = torch::empty({1, words}, bitmask.options().device(device));
        }
        reusable_bitmask_gpu_.copy_(bitmask, /*non_blocking=*/true);
        state.packed_bitmask = reusable_bitmask_gpu_.narrow(1, 0, words);
        state.mode           = DeviceMaskMode::MASK;
        state.ready_event    = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
        state.ready_event->record(cuda_graph::graphGetCurrentStream());
        return state;
    }
#endif

    if (!reusable_vocab_mask_cpu_.defined() || reusable_vocab_mask_cpu_.size(0) < grammar_vocab_size) {
        auto mask_options = torch::TensorOptions().dtype(torch::kBool);
        reusable_vocab_mask_cpu_ = torch::empty({grammar_vocab_size}, mask_options);
    }
    auto           vocab_mask  = reusable_vocab_mask_cpu_.narrow(0, 0, grammar_vocab_size);
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
        case DeviceMaskMode::PASSTHROUGH:
            return;
        case DeviceMaskMode::TERMINATED:
            forceToken(logits, eos_token_id_);
            return;
        case DeviceMaskMode::MASK:
            break;
    }

#if USING_CUDA || USING_ROCM
    if (state.packed_bitmask.defined() && logits.is_cuda()) {
        if (state.ready_event) {
            state.ready_event->block(cuda_graph::graphGetCurrentStream());
        }
        auto logits_2d = logits.unsqueeze(0);
        invokeApplyXGrammarBitmaskInplace(logits_2d,
                                          state.packed_bitmask,
                                          static_cast<int64_t>(state.grammar_vocab_size),
                                          cuda_graph::graphGetCurrentStream().stream());
        return;
    }
#endif

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
        const auto* finished = inputs.finished_mask.data_ptr<bool>();
        if (finished[start_idx]) {
            return;
        }
    }

    auto logits = inputs.logits.narrow(0, start_idx, 1);
    auto state  = getDeviceMaskState(logits.device());
    applyDeviceMaskState(logits[0], state);
}

void GrammarLogitsProcessor::updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
    if (!matcher_ || matcher_->finished()) {
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
    RTP_LLM_LOG_DEBUG("[grammar_commit_trace] commit start: stream=%ld num_accepted=%ld "
                      "num_new_tokens=%d batch=%d",
                      trace_stream_id,
                      matcher_->numAcceptedTokens(),
                      num_new_tokens,
                      batch_size);

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        // dsv4-style commit walk: terminated + EOS -> markFinished and stop;
        // any other token after terminated -> error. Pre-terminal accept failures
        // are handled by advanceMatcher's reportError path.
        bool stop_after_eos = false;
        bool report_after_terminal = false;
        std::vector<int32_t> active_tokens;
        active_tokens.reserve(tokens.size());
        for (int32_t tok : tokens) {
            if (matcher_ && matcher_->isTerminated()) {
                if (tok == static_cast<int32_t>(eos_token_id_)) {
                    matcher_->markFinished();
                    stop_after_eos = true;
                } else {
                    matcher_->markFinished();
                    report_after_terminal = true;
                }
                break;
            }
            active_tokens.push_back(tok);
        }
        if (report_after_terminal) {
            // Drop straight to error reporting. matcher_ is already markFinished so
            // any waiter on state_cv_ takes the FINISHED early-return.
            state_cv_.notify_all();
            reportErrorViaReporter(
                ErrorCode::INVALID_PARAMS,
                "grammar received non-EOS token after terminal state",
                /*stream_lock_held=*/true);
            return;
        }
        if (!stop_after_eos && !advanceMatcher(active_tokens, MatcherAdvance::Commit)) {
            state_cv_.notify_all();
            return;
        }
        syncAcceptedTokenLenLocked();
        if (stop_after_eos) {
            // EOS that consumed the terminal state was not forwarded to the
            // xgrammar matcher (markFinished only), so numAcceptedTokens does
            // not include it. Keep accepted_token_len_ in sync with the
            // stream's committed token count so any pending async waiter sees
            // the EOS as accepted and unblocks immediately.
            ++accepted_token_len_;
        }
        pending_async_token_len_ = accepted_token_len_;
        rebuildDeviceMaskStateLocked();
        state_cv_.notify_all();
    }

    if (auto stream = stream_.lock()) {
        RTP_LLM_LOG_DEBUG("[sp_accept_trace] stream_id=%ld grammar=%s propose_step=%zu accept_len=%d",
                          stream->streamId(),
                          grammarKindTag(stream).c_str(),
                          stream->getProposeStep(),
                          num_new_tokens);
    }
}

bool GrammarLogitsProcessor::isGrammarMatcherComplete() const noexcept {
    if (!matcher_) {
        return false;
    }
    if (matcher_->isTerminated() || matcher_->finished()) {
        return true;
    }
    return matcher_->onlyStopTokenLegalNext(static_cast<int32_t>(eos_token_id_));
}

void GrammarLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& /* src_batch_indices */) {}

bool GrammarLogitsProcessor::isSpecVerifyEligible() const {
    return matcher_ != nullptr && !reported_error_.load(std::memory_order_relaxed);
}

int GrammarLogitsProcessor::tryAcceptAndFillBitmask(const SpecLogitsProcessorRequest& request) {
    if (!matcher_ || request.propose_step <= 0 || request.bitmask_cpu_out == nullptr) {
        return request.propose_step;
    }
    if (reported_error_.load(std::memory_order_relaxed)) {
        // Defensive: commit path already failed for this stream. isSpecVerifyEligible
        // gates this normally, but short-circuit here too in case a caller reaches
        // verify without first consulting it.
        return 0;
    }
    if (request.bitmask_size_int32 < static_cast<size_t>((request.vocab_size + 31) / 32)) {
        // Caller buffer is too small for the model vocab — programming error in
        // the spec executor. Don't abort the whole process; report and skip.
        RTP_LLM_LOG_WARNING("[grammar] tryAcceptAndFillBitmask: bitmask buffer too small "
                            "(words=%zu vocab=%zu); skipping verify",
                            request.bitmask_size_int32, request.vocab_size);
        reported_error_.store(true, std::memory_order_relaxed);
        reportErrorViaReporter(ErrorCode::EXECUTION_EXCEPTION,
                               "grammar MTP verify: bitmask buffer smaller than model vocab",
                               /*stream_lock_held=*/false);
        return 0;
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (isGrammarMatcherComplete()) {
        return 0;
    }

    const int  P               = request.propose_step;
    const auto W               = request.bitmask_size_int32;
    int        accepted_prefix = 0;
    int        cap             = P;

    // Guard against grammar vocab exceeding the model bitmask once up front.
    // Previously this was a hard RTP_LLM_CHECK inside fill_row that aborted the
    // process; degrade gracefully by erroring the stream and skipping verify.
    {
        const int32_t grammar_vocab_size = matcher_->vocabSize();
        if (grammar_vocab_size > 0
            && SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size) > W) {
            RTP_LLM_LOG_WARNING("[grammar] tryAcceptAndFillBitmask: grammar vocab (%d) exceeds "
                                "model vocab bitmask (%zu words); reporting stream error",
                                grammar_vocab_size, W);
            reported_error_.store(true, std::memory_order_relaxed);
            matcher_->markFinished();
            reportErrorViaReporter(
                ErrorCode::EXECUTION_EXCEPTION,
                "grammar vocab exceeds model vocab in MTP verify (grammar=" + std::to_string(grammar_vocab_size)
                    + ", model_words=" + std::to_string(W) + ")",
                /*stream_lock_held=*/false);
            return 0;
        }
    }

    auto fill_row = [&](int32_t* row) -> VerifyRowState {
        std::fill_n(row, W, SpecLogitsProcessor::kBitmaskAllowAll);
        if (matcher_->finished()) {
            return VerifyRowState::Finished;
        }
        if (matcher_->isTerminated()) {
            forceTokenInBitmask(row, W, eos_token_id_);
            return VerifyRowState::Terminated;
        }

        const int32_t grammar_vocab_size = matcher_->vocabSize();
        const size_t  grammar_words      = SpecLogitsProcessor::bitmaskWordCount(grammar_vocab_size);

        int64_t  dl_shape[2];
        DLTensor dl = makeSingleRowBitmaskView(row, static_cast<int32_t>(grammar_words), dl_shape);
        if (!matcher_->fillBitmask(&dl, 0)) {
            return VerifyRowState::Active;
        }
        clearBitmaskTokenRange(row, W, grammar_vocab_size, static_cast<int64_t>(request.vocab_size));
        return VerifyRowState::Active;
    };

    try {
        for (int offset = 0; offset <= P; ++offset) {
            int32_t* row = request.bitmask_cpu_out + offset * W;
            const VerifyRowState row_state = fill_row(row);
            if (offset == P) {
                break;
            }
            if (row_state == VerifyRowState::Terminated || row_state == VerifyRowState::Finished) {
                cap = offset;
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
            // Do NOT early-break on terminated / finished / only-stop-legal here:
            // doing so would leave row[offset+1] holding the prior all-allow contents.
            // Aligns with dsv4 baseline — let the next iteration's fill_row observe
            // the matcher's new state and emit the correct EOS-only / finished row.
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
