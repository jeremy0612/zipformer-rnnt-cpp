#pragma once

#include <string>
#include <vector>
#include <memory>

namespace zipformer {

class BPETokenizer {
public:
    explicit BPETokenizer(const std::string& model_path);
    ~BPETokenizer();

    std::string decode(const std::vector<int>& token_ids) const;
    std::vector<int> encode(const std::string& text) const;

    int blank_id()  const;  // typically 0
    int vocab_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace zipformer
