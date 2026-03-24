#include "bpe_tokenizer.hpp"
#include <sentencepiece_processor.h>
#include <stdexcept>

namespace zipformer {

struct BPETokenizer::Impl {
    sentencepiece::SentencePieceProcessor sp;
};

BPETokenizer::BPETokenizer(const std::string& model_path) : impl_(new Impl) {
    auto status = impl_->sp.Load(model_path);
    if (!status.ok())
        throw std::runtime_error("Failed to load BPE model: " + model_path);
}

BPETokenizer::~BPETokenizer() = default;

std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    impl_->sp.Decode(ids, &out);
    return out;
}

std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    impl_->sp.Encode(text, &ids);
    return ids;
}

int BPETokenizer::blank_id()   const { return 0; }
int BPETokenizer::vocab_size() const { return impl_->sp.GetPieceSize(); }

} // namespace zipformer
