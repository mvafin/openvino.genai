// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/core/model.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "visual_language/processor_config.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {
class VisionEncoder;
class InputsEmbedder;
struct GGUFMultimodalModels {
    std::shared_ptr<ov::Model> language, text_embeddings, vision, audio;
    Tokenizer tokenizer;
    VLMConfig config;
    ProcessorConfig processor;
};

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties);
std::shared_ptr<VisionEncoder> create_gguf_vision_encoder(const GGUFMultimodalModels& models,
                                                          const std::string& device,
                                                          const ov::AnyMap& properties);
void attach_gguf_audio_encoder(InputsEmbedder& embedder,
                               const GGUFMultimodalModels& models,
                               const std::string& device,
                               const ov::AnyMap& properties);
}  // namespace ov::genai
