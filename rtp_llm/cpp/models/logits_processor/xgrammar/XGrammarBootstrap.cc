#include "rtp_llm/cpp/models/logits_processor/xgrammar/XGrammarBootstrap.h"

#include <pybind11/stl.h>
#include <xgrammar/tokenizer_info.h>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/utils/Logger.h"

namespace py = pybind11;

namespace rtp_llm {

namespace {

bool isGrammarBackendDisabled(const std::string& backend) {
    std::string s = backend;
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char c) { return !std::isspace(c); }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char c) { return !std::isspace(c); }).base(), s.end());
    for (auto& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s.empty() || s == "none";
}

std::vector<int32_t> collectStopTokenIds(py::object model) {
    py::object st = model.attr("model_config").attr("special_tokens");
    std::vector<int32_t> ids;
    auto add_id = [&](int32_t id) {
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) {
            ids.push_back(id);
        }
    };

    py::object eos = st.attr("eos_token_id");
    if (py::isinstance<py::int_>(eos)) {
        add_id(eos.cast<int32_t>());
    } else if (py::isinstance<py::list>(eos) || py::isinstance<py::tuple>(eos)) {
        for (py::handle item : eos) {
            add_id(py::reinterpret_borrow<py::object>(item).cast<int32_t>());
        }
    }

    py::object stop_list = st.attr("stop_words_id_list");
    if (!stop_list.is_none()) {
        for (py::handle seq_h : stop_list) {
            py::object seq = py::reinterpret_borrow<py::object>(seq_h);
            if (py::len(seq) == 1) {
                add_id(seq[0].cast<int32_t>());
            }
        }
    }

    std::sort(ids.begin(), ids.end());
    return ids;
}

int64_t resolveThinkTokenId(const std::string& reasoning_parser, py::object tokenizer, const char* token_attr) {
    RTP_LLM_LOG_INFO("[bootstrap-dbg] resolveThinkTokenId enter: parser='%s' attr='%s'",
                     reasoning_parser.c_str(),
                     token_attr ? token_attr : "(null)");
    if (reasoning_parser.empty()) {
        RTP_LLM_LOG_INFO("[bootstrap-dbg] empty parser -> -1");
        return -1;
    }
    if (!token_attr) {
        RTP_LLM_LOG_INFO("[bootstrap-dbg] null token_attr -> -1");
        return -1;
    }
    try {
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step1 import module");
        py::module reasoning_parser_mod =
            py::module::import("rtp_llm.openai.renderers.sglang_helpers.reasoning_parser");
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step2 get ReasoningParser attr");
        py::object parser_cls = reasoning_parser_mod.attr("ReasoningParser");
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step3 build py::str from parser name (len=%zu)", reasoning_parser.size());
        py::str    parser_name(reasoning_parser);
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step4 instantiate ReasoningParser");
        py::object parser = parser_cls(parser_name);
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step5 detector attr");
        py::object detector = parser.attr("detector");
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step6 detector.%s", token_attr);
        py::object token = detector.attr(token_attr);
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step7 token type=%s repr=%s",
                         py::str(token.get_type()).cast<std::string>().c_str(),
                         py::repr(token).cast<std::string>().c_str());
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step8 tokenizer.encode(token)");
        py::object encoded = tokenizer.attr("encode")(token);
        const auto encoded_len = py::len(encoded);
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step9 encode done; len=%zd encoded_type=%s",
                         static_cast<ssize_t>(encoded_len),
                         std::string(py::str(py::type::of(encoded))).c_str());
        if (encoded_len != 1) {
            RTP_LLM_LOG_INFO("[bootstrap-dbg] step9b encoded_len != 1");
            return -1;
        }
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step9c about to fetch encoded[0]");
        py::object first = encoded[py::int_(0)];
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step9d fetched first; type=%s",
                         std::string(py::str(py::type::of(first))).c_str());
        int64_t id = first.cast<int64_t>();
        RTP_LLM_LOG_INFO("[bootstrap-dbg] step10 done id=%lld", static_cast<long long>(id));
        return id;
    } catch (const std::exception& e) {
        RTP_LLM_LOG_WARNING("[bootstrap-dbg] resolveThinkTokenId(%s) caught: %s", token_attr, e.what());
        return -1;
    }
}

}  // namespace

std::string buildXGrammarTokenizerInfoJson(
    const std::unordered_map<std::string, int32_t>& vocab,
    const std::string&                              backend_tokenizer_str,
    int64_t                                         vocab_size,
    const std::vector<int32_t>&                     stop_token_ids) {
    static constexpr int64_t kMaxVocabSize = 1'000'000;
    if (vocab_size <= 0 || vocab_size > kMaxVocabSize) {
        throw std::invalid_argument("vocab_size must be in (0, " + std::to_string(kMaxVocabSize) + "], got "
                                    + std::to_string(vocab_size));
    }
    std::vector<std::string> encoded_vocab(static_cast<size_t>(vocab_size));
    std::vector<bool>        slot_filled(static_cast<size_t>(vocab_size), false);
    for (const auto& [tok, tid] : vocab) {
        if (tid < 0 || tid >= vocab_size) {
            throw std::invalid_argument("token id " + std::to_string(tid) + " is out of vocab range [0, "
                                        + std::to_string(vocab_size) + ")");
        }
        const auto idx = static_cast<size_t>(tid);
        encoded_vocab[idx] = tok;
        slot_filled[idx]   = true;
    }
    // HF tokenizer.get_vocab() can be sparse vs vocab_size when vocab_size is
    // padded above the highest assigned id (e.g. embedding tables sized to a
    // multiple of 32). Empty slots reach xgrammar as zero-length tokens, where
    // they are accepted as legal-but-empty matches and let the DFA cross
    // grammar boundaries on what should be unreachable ids. Count and warn so
    // the malformation is visible rather than masked.
    size_t empty_slots = 0;
    for (size_t i = 0; i < slot_filled.size(); ++i) {
        if (!slot_filled[i]) {
            ++empty_slots;
        }
    }
    if (empty_slots > 0) {
        RTP_LLM_LOG_WARNING(
            "buildXGrammarTokenizerInfoJson: %zu/%lld vocab slots not covered by tokenizer.get_vocab(); "
            "grammar may admit zero-length matches at those ids",
            empty_slots,
            static_cast<long long>(vocab_size));
    }

    std::string meta  = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);
    std::string stops = "[";
    for (size_t i = 0; i < stop_token_ids.size(); ++i) {
        if (i) stops += ",";
        stops += std::to_string(stop_token_ids[i]);
    }
    stops += "]";

    const auto close_pos = meta.rfind('}');
    if (close_pos == std::string::npos) {
        throw std::runtime_error(
            "buildXGrammarTokenizerInfoJson: DetectMetadataFromHF returned a non-object metadata string: '"
            + meta + "'");
    }
    const auto first_content = meta.find_first_not_of(" \t\r\n", meta.find('{') + 1);
    const bool has_members   = first_content != std::string::npos && first_content < close_pos;
    std::string injected = std::string(has_members ? "," : "") + "\"vocab_size\":" + std::to_string(vocab_size)
                           + ",\"stop_token_ids\":" + stops;
    meta.insert(close_pos, injected);
    return xgrammar::TokenizerInfo::FromVocabAndMetadata(encoded_vocab, meta).SerializeJSON();
}

void bootstrapGrammarConfigFromModel(py::object model, GrammarConfig& gc, ReasoningConfig& rc) {
    RTP_LLM_LOG_INFO("[bootstrap-dbg] enter; backend='%s' reasoning_parser='%s'",
                     gc.grammar_backend.c_str(),
                     rc.reasoning_parser.c_str());
    if (isGrammarBackendDisabled(gc.grammar_backend)) {
        RTP_LLM_LOG_INFO("[bootstrap-dbg] backend disabled, returning");
        gc.tokenizer_info_json = "";
        return;
    }

    std::vector<int32_t> stop_token_ids;
    int64_t              vocab_size = 0;
    try {
        RTP_LLM_LOG_INFO("[bootstrap-dbg] B1 model.tokenizer.tokenizer");
        py::object tokenizer = model.attr("tokenizer").attr("tokenizer");
        RTP_LLM_LOG_INFO("[bootstrap-dbg] B2 get_vocab");
        auto       vocab     = tokenizer.attr("get_vocab")().cast<std::unordered_map<std::string, int32_t>>();
        RTP_LLM_LOG_INFO("[bootstrap-dbg] B3 vocab_size attr");
        vocab_size           = model.attr("model_config").attr("vocab_size").cast<int64_t>();
        if (!vocab.empty()) {
            int32_t max_id = 0;
            for (const auto& [_, tid] : vocab) {
                max_id = std::max(max_id, tid);
            }
            vocab_size = std::max(vocab_size, static_cast<int64_t>(max_id) + 1);
        }
        RTP_LLM_LOG_INFO("[bootstrap-dbg] B4 collectStopTokenIds");
        stop_token_ids = collectStopTokenIds(model);
        RTP_LLM_LOG_INFO("[bootstrap-dbg] B5 backend_tokenizer.to_str");
        const std::string backend_str =
            tokenizer.attr("backend_tokenizer").attr("to_str")().cast<std::string>();
        RTP_LLM_LOG_INFO("[bootstrap-dbg] B6 buildXGrammarTokenizerInfoJson vocab_size=%lld",
                         static_cast<long long>(vocab_size));
        gc.tokenizer_info_json = buildXGrammarTokenizerInfoJson(vocab, backend_str, vocab_size, stop_token_ids);
        RTP_LLM_LOG_INFO("[bootstrap-dbg] B7 json built (%zuB)", gc.tokenizer_info_json.size());
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("xgrammar bootstrap failed (%s); grammar disabled", e.what());
        gc.tokenizer_info_json = "";
        return;
    }

    if (!rc.reasoning_parser.empty()) {
        RTP_LLM_LOG_INFO("[bootstrap-dbg] R0 resolving think tokens; parser='%s'",
                         rc.reasoning_parser.c_str());
        py::object tokenizer = model.attr("tokenizer").attr("tokenizer");
        RTP_LLM_LOG_INFO("[bootstrap-dbg] R1 think_end");
        rc.think_end_id      = resolveThinkTokenId(rc.reasoning_parser, tokenizer, "think_end_token");
        RTP_LLM_LOG_INFO("[bootstrap-dbg] R2 think_start");
        rc.think_start_id    = resolveThinkTokenId(rc.reasoning_parser, tokenizer, "think_start_token");
        RTP_LLM_LOG_INFO("[bootstrap-dbg] R3 done end=%lld start=%lld",
                         static_cast<long long>(rc.think_end_id),
                         static_cast<long long>(rc.think_start_id));
    } else {
        RTP_LLM_LOG_INFO("[bootstrap-dbg] reasoning_parser empty, skipping think token resolution");
    }

    RTP_LLM_LOG_INFO(
        "grammar bootstrap: vocab=%lld, json=%zuB, think_end_id=%lld, think_start_id=%lld, stop_tokens=%zu",
        static_cast<long long>(vocab_size),
        gc.tokenizer_info_json.size(),
        static_cast<long long>(rc.think_end_id),
        static_cast<long long>(rc.think_start_id),
        stop_token_ids.size());
}

}  // namespace rtp_llm
