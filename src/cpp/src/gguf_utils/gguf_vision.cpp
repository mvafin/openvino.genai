// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>

#include "gguf_multimodal.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/gemma3/classes.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/qwen2vl/classes.hpp"

namespace ov::genai {
namespace {
class GGUFGemma4VisionEncoder : public VisionEncoderGemma4 {
public:
    GGUFGemma4VisionEncoder(const std::shared_ptr<ov::Model>& model,
                            const ProcessorConfig& processor,
                            const std::string& device,
                            const ov::AnyMap& properties)
        : VisionEncoderGemma4(model, processor, device, properties) {
        static_cast<ProcessorConfig&>(m_video_processor_config) = processor;
    }

protected:
    EncodedImage encode_with_config(const ov::Tensor& image, const ProcessorConfig& config) override {
        const auto source = tensor_to_clip_image_u8(image);
        const auto size = qwen2_vl_utils::smart_resize(source.ny,
                                                       source.nx,
                                                       config.patch_size * config.merge_size,
                                                       config.min_pixels,
                                                       config.max_pixels);
        clip_image_u8 resized;
        bicubic_resize(source, resized, size.width, size.height);
        clip_ctx ctx;
        std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
        std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
        auto normalized = clip_image_preprocess(ctx, resized);
        ov::Tensor pixels(ov::element::f32, {1, 3, size.height, size.width}, normalized.buf.data());
        const auto gh = size.height / config.patch_size, gw = size.width / config.patch_size;
        ov::Tensor x(ov::element::i32, {1, 1, 1, gh * gw});
        ov::Tensor y(ov::element::i32, x.get_shape());
        for (size_t i = 0; i < gh * gw; ++i) {
            x.data<int32_t>()[i] = i % gw;
            y.data<int32_t>()[i] = i / gw;
        }
        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
        auto& request = guard.get();
        request.set_tensor("pixel_values", pixels);
        request.set_tensor("position_x", x);
        request.set_tensor("position_y", y);
        request.infer();
        const auto output = request.get_tensor("image_features");
        ov::Tensor features(output.get_element_type(), output.get_shape());
        output.copy_to(features);
        return {std::move(features)};
    }
};

// GGUF includes patch embedding, position interpolation, the encoder and merger.
// Its inputs therefore remain normalized pixels and explicit patch/position indices.
class GGUFQwenVisionEncoder : public VisionEncoder {
public:
    GGUFQwenVisionEncoder(const std::shared_ptr<ov::Model>& model,
                          const ProcessorConfig& processor,
                          const std::string& device,
                          const ov::AnyMap& properties)
        : VisionEncoder(model, processor, device, properties) {
        static_cast<ProcessorConfig&>(m_video_processor_config) = processor;
        m_video_processor_config.fps = 2.f;
    }

    EncodedImage encode(const ov::Tensor& image, const ov::AnyMap& properties) override {
        return encode_pair(image, image, ProcessorConfig::from_any_map(properties, m_processor_config));
    }

    EncodedVideo encode_frames(const std::vector<ov::Tensor>& frames) override {
        OPENVINO_ASSERT(!frames.empty(), "Qwen video requires at least one frame");
        EncodedVideo result;
        std::vector<ov::Tensor> features;
        for (size_t i = 0; i < frames.size(); i += 2) {
            auto pair = encode_pair(frames[i], frames[std::min(i + 1, frames.size() - 1)], m_video_processor_config);
            if (!features.empty()) {
                OPENVINO_ASSERT(result.resized_source_size.height == pair.resized_source_size.height &&
                                    result.resized_source_size.width == pair.resized_source_size.width,
                                "Qwen video frames must have equal resized grids");
            }
            result.resized_source_size = pair.resized_source_size;
            result.num_video_tokens += pair.num_image_tokens;
            features.push_back(std::move(pair.resized_source));
        }
        result.frame_num = features.size();
        result.video_features = qwen2_vl_utils::concatenate_video_image_embeds(features, {});
        return result;
    }

private:
    EncodedImage encode_pair(const ov::Tensor& first, const ov::Tensor& second, const ProcessorConfig& config) {
        OPENVINO_ASSERT(config.temporal_patch_size == 2, "GGUF Qwen requires temporal_patch_size=2");
        const auto& shape = first.get_shape();
        const auto size = qwen2_vl_utils::smart_resize(shape.at(1),
                                                       shape.at(2),
                                                       config.patch_size * config.merge_size,
                                                       config.min_pixels,
                                                       config.max_pixels);
        ov::Tensor pixels(ov::element::f32, {2, 3, size.height, size.width});
        clip_ctx ctx;
        std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
        std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
        const ov::Tensor frames[] = {first, second};
        for (size_t t = 0; t < 2; ++t) {
            clip_image_u8 resized;
            bicubic_resize(tensor_to_clip_image_u8(frames[t]), resized, size.width, size.height);
            const auto normalized = clip_image_preprocess(ctx, resized);
            std::memcpy(pixels.data<float>() + t * normalized.buf.size(),
                        normalized.buf.data(),
                        normalized.buf.size() * sizeof(float));
        }
        const size_t gh = size.height / config.patch_size, gw = size.width / config.patch_size;
        const size_t count = gh * gw, merge = config.merge_size;
        ov::Tensor indices(ov::element::i32, {1, 1, 1, count});
        ov::Tensor positions(ov::element::i32, {1, 1, 1, 4 * count});
        size_t i = 0;
        for (size_t y = 0; y < gh; y += merge) {
            for (size_t x = 0; x < gw; x += merge) {
                for (size_t dy = 0; dy < merge; ++dy) {
                    for (size_t dx = 0; dx < merge; ++dx, ++i) {
                        indices.data<int32_t>()[i] = (y + dy) * gw + x + dx;
                        positions.data<int32_t>()[i] = positions.data<int32_t>()[2 * count + i] = y + dy;
                        positions.data<int32_t>()[count + i] = positions.data<int32_t>()[3 * count + i] = x + dx;
                    }
                }
            }
        }
        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
        auto& request = guard.get();
        request.set_tensor("pixel_values", pixels);
        request.set_tensor("patch_indices", indices);
        request.set_tensor("position_ids", positions);
        request.infer();
        const auto output = request.get_tensor("image_features");
        const auto& output_shape = output.get_shape();
        EncodedImage result;
        result.resized_source = ov::Tensor(output.get_element_type(), {output_shape.at(1), output_shape.at(2)});
        std::memcpy(result.resized_source.data(), output.data(), output.get_byte_size());
        result.resized_source_size = {gh, gw};
        result.num_image_tokens = output_shape[1];
        return result;
    }
};
}  // namespace

std::shared_ptr<VisionEncoder> create_gguf_vision_encoder(const GGUFMultimodalModels& models,
                                                          const std::string& device,
                                                          const ov::AnyMap& properties) {
    switch (models.config.model_type) {
    case VLMModelType::GEMMA3:
        return std::make_shared<VisionEncoderGemma3>(models.vision, models.processor, device, properties);
    case VLMModelType::GEMMA4:
    case VLMModelType::GEMMA4_UNIFIED:
        return std::make_shared<GGUFGemma4VisionEncoder>(models.vision, models.processor, device, properties);
    case VLMModelType::MUSE_GLIMMER:
        return std::make_shared<VisionEncoderMuseGlimmer>(models.vision, models.processor, device, properties);
    case VLMModelType::QWEN3_5:
    case VLMModelType::QWEN3_5_MOE:
        return std::make_shared<GGUFQwenVisionEncoder>(models.vision, models.processor, device, properties);
    default:
        OPENVINO_THROW("Unsupported GGUF vision encoder family");
    }
}
}  // namespace ov::genai
