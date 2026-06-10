#include "rtp_llm/cpp/models/logits_processor/xgrammar/pybind/XGrammarBootstrap.h"

#include <pybind11/stl.h>
#include <xgrammar/tokenizer_info.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace rtp_llm {

// Take the vocab as a dict<token_str, token_id> straight from
// tokenizer.get_vocab() and do the "id -> token" reverse fill in C++. Saves
// the Python-side `[""] * vocab_size` allocation + per-id assignment loop;
// xgrammar's FromVocabAndMetadata still needs the indexed vector internally.
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

    // DetectMetadataFromHF returns {"vocab_type":N,"add_prefix_space":bool};
    // splice in vocab_size + stop_token_ids before the closing '}' rather
    // than pulling in a JSON lib just to rebuild 4 keys.
    std::string meta  = xgrammar::TokenizerInfo::DetectMetadataFromHF(backend_tokenizer_str);
    std::string stops = "[";
    for (size_t i = 0; i < stop_token_ids.size(); ++i) {
        if (i) stops += ",";
        stops += std::to_string(stop_token_ids[i]);
    }
    stops += "]";

    // Insert before the final '}'. Guard against a malformed metadata string:
    // an empty meta would underflow meta.size()-1, and a bare "{}" would yield
    // a leading-comma ("{,...}") that is invalid JSON. Locate '}' explicitly
    // and only prepend a comma when the object already has members.
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

void registerXGrammarBootstrap(py::module& m) {
    m.def("build_xgrammar_tokenizer_info_json",
          &buildXGrammarTokenizerInfoJson,
          py::arg("vocab"),
          py::arg("backend_tokenizer_str"),
          py::arg("vocab_size"),
          py::arg("stop_token_ids"));
}

}  // namespace rtp_llm
