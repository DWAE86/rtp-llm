#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <xgrammar/compiler.h>

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBackend.h"

namespace rtp_llm {

class RtpGrammarMatcher;

// Result of one async grammar compile.
//   compiled != null -> success (cache_hit tells whether it was served from the
//                       backend cache without a fresh compile).
//   is_invalid       -> cacheable schema rejection.
//   otherwise        -> system-level error (not cached, retry next time).
struct GrammarReadyPayload {
    std::shared_ptr<xgrammar::CompiledGrammar> compiled;
    bool                                       is_invalid      = false;
    bool                                       cache_hit       = false;
    std::string                                error_msg;
    int64_t                                    compile_time_us = 0;
};

// Process-level grammar compile service. Owns the XGrammarBackend plus a
// worker pool, dedups concurrent identical compiles via shared_future
// (singleflight), and caches results eagerly. This is the worker-pool half of
// the former GrammarCompileGate; the scheduler-facing pre-admission "gate" is
// gone — per-stream readiness is now handled by the pending GrammarLogitsProcessor,
// which only submits keys here and polls the returned future from prepare().
//
// Disabled mode (backend == nullptr): submit() is never reached because
// GrammarLogitsProcessor::tryCreatePending fails the request before submitting;
// enabled() reports false so callers learn at admission that grammar is unavailable.
class GrammarCompiler {
public:
    // Build the backend + worker pool from cfg. Idempotent on matching config.
    // Throws std::runtime_error on re-init with a DIFFERENT config — silently
    // keeping the first install would route a second model's grammar through
    // the wrong tokenizer. On backend build failure the singleton stays
    // disabled (enabled() == false) so grammar requests are rejected early
    // with a clear message. Registration of the logits-processor factory is
    // done by the engine, not here, to keep the compiler free of any
    // dependency on GrammarLogitsProcessor.
    static void             initialize(const GrammarConfig& cfg);
    static GrammarCompiler& instance();
    static void             resetForTest() noexcept;  // unit-test hook only

    ~GrammarCompiler();

    GrammarCompiler(const GrammarCompiler&)            = delete;
    GrammarCompiler& operator=(const GrammarCompiler&) = delete;

    bool    enabled() const noexcept { return backend_ != nullptr; }
    int64_t compileTimeoutMs() const noexcept { return grammar_compile_timeout_ms_; }
    int64_t maskWaitTimeoutMs() const noexcept { return mask_wait_timeout_ms_; }

    // Submit a compile request. A cache hit / cached-invalid returns an
    // already-ready future; a miss dedups onto (or starts) the in-flight compile
    // for an identical key. Never blocks on the compile itself.
    std::shared_future<GrammarReadyPayload> submit(const GrammarKeyCpp& key);

    // Build a fresh per-stream matcher around an already-compiled grammar.
    // Reasoning / think-body gating is layered on top by the caller via
    // ReasoningGate; the matcher itself is unaware of think semantics.
    std::unique_ptr<RtpGrammarMatcher> createMatcher(std::shared_ptr<xgrammar::CompiledGrammar> compiled,
                                                     bool terminate_without_stop_token = false);

    // Test-only: current depth of the pending-compile queue.
    size_t pendingTasks() const;

private:
    explicit GrammarCompiler(std::shared_ptr<XGrammarBackend> backend, const GrammarConfig& cfg);

    // Translate a GrammarConfig into a live (enabled) backend, or nullptr when
    // structured output is disabled / the backend is unhealthy / the backend
    // name is unknown. Never throws.
    static std::shared_ptr<XGrammarBackend> buildBackend(const GrammarConfig& cfg) noexcept;

    void                                    workerLoop();
    std::shared_future<GrammarReadyPayload> makeReadyFuture(GrammarReadyPayload payload) const;

    struct CompileTask {
        GrammarKeyCpp                     key;
        std::promise<GrammarReadyPayload> promise;
    };

    std::shared_ptr<XGrammarBackend> backend_;

    mutable std::mutex      queue_mutex_;
    std::condition_variable worker_cv_;
    std::deque<CompileTask> compile_tasks_;
    // singleflight: identical concurrent submits share one future. The worker
    // erases the slot once the compile completes, after which later submits hit
    // the backend cache. Keyed by GrammarKeyCpp::id().
    std::unordered_map<std::string, std::shared_future<GrammarReadyPayload>> in_flight_;

    std::vector<std::thread> workers_;
    std::atomic<bool>        stop_{false};
    std::atomic<int>         alive_workers_{0};

    int64_t grammar_compile_timeout_ms_ = 60000;
    int64_t mask_wait_timeout_ms_       = 5000;

    static std::mutex                       singleton_mutex_;
    static std::unique_ptr<GrammarCompiler> singleton_;
    static bool                             initialized_;
    static std::optional<size_t>            config_fingerprint_;
};

}  // namespace rtp_llm
