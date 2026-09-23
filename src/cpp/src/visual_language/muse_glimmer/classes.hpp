// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <mutex>

#include "visual_language/inputs_embedder.hpp"
#include "visual_language/vision_encoder.hpp"
#include "visual_language/vlm_config.hpp"

namespace ov::genai {

class VisionEncoderMuseGlimmer : public VisionEncoder {
public:
    using VisionEncoder::VisionEncoder;
    VisionEncoderMuseGlimmer(const std::shared_ptr<ov::Model>& model,
                             const ProcessorConfig& config,
                             const std::string& device,
                             const ov::AnyMap& properties);

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& config_map = {}) override;

    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override;

private:
    size_t m_gguf_window = 0;
    EncodedImage encode_with_config(const std::vector<ov::Tensor>& frames,
                                    const ProcessorConfig& config,
                                    size_t max_tokens);
};

class InputsEmbedderMuseGlimmer : public InputsEmbedder::IInputsEmbedder {
public:
    InputsEmbedderMuseGlimmer(const VLMConfig& config,
                              const Tokenizer& tokenizer,
                              const VisionEncoder::Ptr& vision,
                              const EmbeddingsModel::Ptr& embeddings,
                              const std::string& device)
        : IInputsEmbedder(config, tokenizer, vision, embeddings, device) {}
    InputsEmbedderMuseGlimmer(const VLMConfig& vlm_config,
                              const std::filesystem::path& model_dir,
                              const Tokenizer& tokenizer,
                              const std::string& device,
                              const ov::AnyMap device_config);

    InputsEmbedderMuseGlimmer(const VLMConfig& vlm_config,
                              const ModelsMap& models_map,
                              const Tokenizer& tokenizer,
                              const std::filesystem::path& config_dir_path,
                              const std::string& device,
                              const ov::AnyMap device_config);

    std::vector<EncodedImage> encode_images(const std::vector<ov::Tensor>& images) override;

    std::vector<EncodedVideo> encode_videos(const std::vector<ov::Tensor>& videos,
                                            const std::vector<VideoMetadata>& videos_metadata = {}) override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_id,
                                      const std::vector<EncodedImage>& images) const override;

    NormalizedPrompt normalize_prompt(const std::string& prompt,
                                      size_t base_image_id,
                                      size_t base_video_id,
                                      const std::vector<EncodedImage>& images,
                                      const std::vector<EncodedVideo>& videos) const override;

    ov::Tensor get_inputs_embeds(const std::string& prompt,
                                 const std::vector<EncodedImage>& images,
                                 VLMPerfMetrics& metrics,
                                 bool recalculate_merged_embeddings = true,
                                 const std::vector<size_t>& image_sequence = {}) override;

    ov::Tensor get_inputs_embeds(
        const std::string& prompt,
        const std::vector<EncodedImage>& images,
        const std::vector<EncodedVideo>& videos,
        VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings = true,
        const std::vector<size_t>& image_sequence = {},
        const std::vector<size_t>& videos_sequence = {},
        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count = {}) override;

private:
    int64_t m_image_token_id = -1;
    int64_t m_video_token_id = -1;
    std::once_flag m_vision_token_ids_once_flag;

    ov::Tensor apply_chat_template_tokenize(const std::string& prompt, VLMPerfMetrics& metrics) override;
    void encode_vision_token_ids();
};

}  // namespace ov::genai
