#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateTypes.h"

namespace rtp_llm {

class GenerateStream;

// Result of polling a logits-processor's asynchronous pre-run preparation.
//   Pending : not ready yet; the owning stream must stay in WAITING and poll again.
//   Ready   : preparation finished and the processor is fully usable.
//   Failed  : preparation failed; the processor has already reported the error on
//             the stream (so the state machine will move it to FINISHED).
enum class PrepareState { Pending, Ready, Failed };

// Logits-processor readiness gate (GenerateStream integration)
//
// Some processors (notably GrammarLogitsProcessor) finish construction before
// their async pre-run work completes (e.g. xgrammar compile). GenerateStream
// arms preparation_pending_ at construction when any attached processor reports
// needsPreparation(). While the stream stays in StreamState::WAITING,
// GenerateStream::pollLogitsProcessorPreparationWhileWaiting() (called from
// moveToNext under mutex_) polls every processor's prepare() each scheduler tick:
//   Pending -> stay in WAITING (poll again next tick)
//   Ready   -> processor installed its resource (e.g. grammar matcher)
//   Failed  -> processor already reported the error on the stream (without-lock
//              variant; moveToNext holds mutex_) so the state machine moves to
//              FINISHED on a subsequent tick once the Error event is visible
//
// This mirrors how LOADING_CACHE polls loadCacheDone(). KV-cache allocation runs
// in LOADING_CACHE (a different status), so a pending grammar compile parks the
// stream in WAITING without holding KV blocks.
//
// Scheduler coupling (FIFOScheduler two-phase WAITING handling):
//   Phase 1 (evaluateWaitingStreams): skip streams while isPreparationPending()
//       && !CanRun, so CanRun and memory checks are not granted until every
//       processor resolves.
//   Phase 2 (evaluateAndUpdateStreams): calls moveToNext(), which runs the gate.
// When all preparations resolve (Ready or Failed), the gate clears
// preparation_pending_ but still returns WAITING for that tick so Phase 1 can
// run evaluateRunningMemory on the NEXT tick. Falling through to
// generate_status_->moveToNext() in the same tick would bypass that check and
// could mix a freshly-ready prefill stream into an ongoing decode batch.
//
// DO NOT gate pollLogitsProcessorPreparationWhileWaiting() on LoadInitiated.
// On the normal (non-PD) path evaluateWaitingStreams skips a stream while
// isPreparationPending() && !CanRun, so CanRun is never granted until
// preparation resolves. Gating on LoadInitiated would deadlock: the scheduler
// waits for preparation, the gate waits for LoadInitiated, and prepare() is
// never polled.
//
// PD-decode: DecodeRpcServer::allocateResource busy-waits moveToNext() until
// !isPreparationPending() (matcher installed) before enqueue, so a PD stream
// never reaches RUNNING with an unresolved processor.

// BaseLogitsProcessor is the interface every logits-processor implements: a
// per-batch process(), per-stream status-update hooks, and an optional
// metrics-reporting hook. GrammarLogitsProcessor extends it with the
// vocabulary-restricting grammar mask plus whole-batch spec-decoding hooks (see
// GrammarLogitsProcessor.h); pure bias processors (ThinkMode, MultiSeq, ...)
// implement only this base.
class BaseLogitsProcessor {
public:
    BaseLogitsProcessor() = default;
    virtual ~BaseLogitsProcessor() {}
    static const float neg_inf;

public:
    virtual void process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) = 0;
    virtual void processSpeculative(const SamplerInputs&        inputs,
                                    size_t                      start_idx,
                                    size_t                      finish_idx,
                                    const std::vector<int32_t>& draft_prefix) {
        (void)draft_prefix;
        process(inputs, start_idx, finish_idx);
    }
    virtual void updateMultiSeqStatus(const std::vector<int>& src_batch_indices)           = 0;

    // Optional asynchronous pre-run preparation (e.g. grammar compile). See the
    // "Logits-processor readiness gate" block above for scheduler/PD coupling.
    // Overrides: needsPreparation() while incomplete; prepare() polled under
    // mutex_ until Pending clears. Ready => fully installed; Failed => already
    // reported error on stream. Idempotent once resolved.
    virtual bool         needsPreparation() const { return false; }
    virtual PrepareState prepare(GenerateStream& /*stream*/) { return PrepareState::Ready; }

    virtual bool isStateful() const { return false; }
    virtual bool supportsNormalAsyncDeviceState() const { return false; }
    virtual void prepareNormalAsyncUpdate(const torch::Tensor& new_tokens, int32_t num_new_tokens) {
        (void)new_tokens;
        (void)num_new_tokens;
    }
    virtual int64_t acceptedTokenLen() const { return 0; }

    // Set the owning stream's back-reference. Called by GenerateStream after
    // make_shared so shared_from_this is valid. Default no-op for processors
    // that don't need a stream reference.
    virtual void setStream(const std::shared_ptr<GenerateStream>& /*stream*/) {}

    // ENGINE INVARIANT: updateStatus is called exactly once for every token
    // (or per-stream commit batch) that GenerateStream commits to the
    // stream's output, regardless of which path produced it:
    //   * normal decode      -> GenerateStream::update()
    //   * spec / MTP prefill -> GenerateStream::specUpdate() (prefill T0)
    //   * spec / MTP decode  -> GenerateStream::specUpdate() (accept suffix)
    //   * disagg replay      -> GenerateStream::update()
    // Stateful processors (GrammarLogitsProcessor, ThinkMode counter,
    // ...) advance their internal state here.
    //
    // Spec executors / schedulers / disagg paths MUST NOT bypass this hook
    // with their own per-processor commit calls — doing so will double-
    // commit on decode and miss prefill T0 (see the 2026-05 xgrammar MTP
    // T0 regression for the historical motivation).
    virtual void updateStatus(const torch::Tensor& new_tokens, int32_t num_new_tokens) = 0;

    // PD-decode "PREBUILT" replay hook: synchronize processor state with token(s)
    // that were already sampled on the prefill side and merely transferred over.
    // Unlike updateStatus, this MUST NOT trigger termination side effects
    // (markFinished / GenerateDone) — the bonus token is not a fresh sampling
    // event; the decode-side first-mask-and-sample step decides termination.
    // Stateless processors leave this as a no-op.
    virtual void replayCommittedTokens(const std::vector<int32_t>& /*tokens*/) {}

    // PD-decode replay follow-up: if grammar already completed during replay
    // (e.g. single-token regex saturation), end the stream now that bonus T0
    // is visible in the output queue. Matches SGLang PREBUILT
    // grammar.finished = req.finished() without treating replay as a fresh commit.
    virtual void finalizeAfterReplayIfComplete() {}

    void          memFill(const torch::Tensor& new_tokens_logits, size_t vocab_size, size_t index);
    void          maskLogits(torch::Tensor& new_token_logits, const torch::Tensor& vocab_mask);
    torch::Tensor generateVocabMask(size_t                                  batch_size,
                                    size_t                                  vocab_size,
                                    const std::vector<std::vector<size_t>>& batch_candidate_token_ids);
};

typedef std::shared_ptr<BaseLogitsProcessor> BaseLogitsProcessorPtr;

}  // namespace rtp_llm
