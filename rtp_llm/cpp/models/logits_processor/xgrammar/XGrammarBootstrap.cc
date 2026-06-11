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
    if (reasoning_parser.empty()) {
        return -1;
    }
    try {
        py::module reasoning_parser_mod =
            py::module::import("rtp_llm.openai.renderers.sglang_helpers.reasoning_parser");
        py::object parser = reasoning_parser_mod.attr("ReasoningParser")(py::arg("model_type") = reasoning_parser);
        py::object token  = parser.attr("detector").attr(token_attr);
        py::object encoded = tokenizer.attr("encode")(token, py::arg("add_special_tokens") = false);
        if (py::len(encoded) != 1) {
            return -1;
        }
        return encoded[0].cast<int64_t>();
    } catch (const std::exception& e) {
        if (std::string(token_attr) == "think_end_token") {
            RTP_LLM_LOG_WARNING("reasoner_grammar: think_end_id resolve failed (%s); reasoner disabled", e.what());
        }
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
    for (const auto& [tok, tid] : vocab) {
        if (tid < 0 || tid >= vocab_size) {
            throw std::invalid_argument("token id " + std::to_string(tid) + " is out of vocab range [0, "
                                        + std::to_string(vocab_size) + ")");
        }
        encoded_vocab[static_cast<size_t>(tid)] = tok;
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
    if (isGrammarBackendDisabled(gc.grammar_backend)) {
        gc.tokenizer_info_json = "";
        return;
    }

    std::vector<int32_t> stop_token_ids;
    int64_t              vocab_size = 0;
    try {
        py::object tokenizer = model.attr("tokenizer").attr("tokenizer");
        auto       vocab     = tokenizer.attr("get_vocab")().cast<std::unordered_map<std::string, int32_t>>();
        vocab_size           = model.attr("model_config").attr("vocab_size").cast<int64_t>();
        if (!vocab.empty()) {
            int32_t max_id = 0;
            for (const auto& [_, tid] : vocab) {
                max_id = std::max(max_id, tid);
            }
            vocab_size = std::max(vocab_size, static_cast<int64_t>(max_id) + 1);
        }
        stop_token_ids = collectStopTokenIds(model);
        const std::string backend_str =
            tokenizer.attr("backend_tokenizer").attr("to_str")().cast<std::string>();
        gc.tokenizer_info_json = buildXGrammarTokenizerInfoJson(vocab, backend_str, vocab_size, stop_token_ids);
    } catch (const std::exception& e) {
        RTP_LLM_LOG_ERROR("xgrammar bootstrap failed (%s); grammar disabled", e.what());
        gc.tokenizer_info_json = "";
        return;
    }

    if (!rc.reasoning_parser.empty()) {
        py::object tokenizer = model.attr("tokenizer").attr("tokenizer");
        rc.think_end_id      = resolveThinkTokenId(rc.reasoning_parser, tokenizer, "think_end_token");
        rc.think_start_id    = resolveThinkTokenId(rc.reasoning_parser, tokenizer, "think_start_token");
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
