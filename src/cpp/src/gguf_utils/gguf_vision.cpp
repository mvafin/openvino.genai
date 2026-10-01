// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "gguf_multimodal.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/gemma3/classes.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/qwen2vl/classes.hpp"

namespace ov::genai {
namespace {
clip_ctx make_clip_ctx(const ProcessorConfig& config) {
    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    return ctx;
}

// Grid indices with each merge x merge window contiguous, windows in row-major order.
std::vector<int32_t> merge_order(size_t height, size_t width, size_t merge) {
    std::vector<int32_t> order;
    order.reserve(height * width);
    for (size_t y = 0; y < height; y += merge)
        for (size_t x = 0; x < width; x += merge)
            for (size_t dy = 0; dy < merge; ++dy)
                for (size_t dx = 0; dx < merge; ++dx)
                    order.push_back(int32_t((y + dy) * width + x + dx));
    return order;
}

ov::Tensor index_tensor(const std::vector<int32_t>& values) {
    ov::Tensor tensor(ov::element::i32, {1, 1, 1, values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int32_t>());
    return tensor;
}

class GGUFGemma4VisionEncoder : public VisionEncoderGemma4 {
public:
    GGUFGemma4VisionEncoder(const std::shared_ptr<ov::Model>& model,
                            const ProcessorConfig& processor,
                            const std::string& device,
                            const ov::AnyMap& properties)
        : VisionEncoderGemma4(model, processor, device, properties) {}

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
        auto ctx = make_clip_ctx(config);
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
        auto ctx = make_clip_ctx(config);
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
        const size_t count = gh * gw;
        const auto order = merge_order(gh, gw, config.merge_size);
        // Four M-RoPE sections: rows, columns, rows, columns.
        ov::Tensor positions(ov::element::i32, {1, 1, 1, 4 * count});
        auto* position = positions.data<int32_t>();
        for (size_t i = 0; i < count; ++i) {
            position[i] = position[2 * count + i] = order[i] / int32_t(gw);
            position[count + i] = position[3 * count + i] = order[i] % int32_t(gw);
        }
        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
        auto& request = guard.get();
        request.set_tensor("pixel_values", pixels);
        request.set_tensor("patch_indices", index_tensor(order));
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

// The GGUF encoder takes NCHW pixels and llama.cpp's window ordering inputs instead of
// patchified pixels and a grid.
class GGUFMuseVisionEncoder : public VisionEncoderMuseGlimmer {
public:
    GGUFMuseVisionEncoder(const std::shared_ptr<ov::Model>& model,
                          const ProcessorConfig& processor,
                          const std::string& device,
                          const ov::AnyMap& properties)
        : VisionEncoderMuseGlimmer(model, processor, device, properties),
          m_window(std::stoull(model->get_rt_info<std::string>({"gguf_mmproj", "vision.window_size"}))) {}

protected:
    void set_encoder_inputs(ov::InferRequest& encoder,
                            const ov::Tensor& pixel_values,
                            const ov::Tensor& image_grid_thw,
                            const ProcessorConfig& config) override {
        const auto* grid = image_grid_thw.data<const int64_t>();
        const size_t gh = grid[1], gw = grid[2], n = gh * gw, patch = config.patch_size;
        // Patches are [n, 3, patch, patch]; restore the [1, 3, H, W] image.
        ov::Tensor pixels(ov::element::f32, {1, 3, gh * patch, gw * patch});
        const auto* flat = pixel_values.data<const float>();
        auto* image = pixels.data<float>();
        for (size_t i = 0; i < n; ++i)
            for (size_t c = 0; c < 3; ++c)
                for (size_t y = 0; y < patch; ++y)
                    std::copy_n(flat + ((i * 3 + c) * patch + y) * patch,
                                patch,
                                image + c * n * patch * patch + (i / gw * patch + y) * gw * patch + i % gw * patch);
        // Window-major patch order; each window attends only to itself.
        std::vector<int32_t> order, inverse(n), xs(n), ys(n);
        std::vector<size_t> window_sizes;
        for (size_t wy = 0; wy < gh; wy += m_window)
            for (size_t wx = 0; wx < gw; wx += m_window) {
                const size_t before = order.size();
                for (size_t y = wy; y < std::min(wy + m_window, gh); ++y)
                    for (size_t x = wx; x < std::min(wx + m_window, gw); ++x)
                        order.push_back(int32_t(y * gw + x));
                window_sizes.push_back(order.size() - before);
            }
        for (size_t i = 0; i < n; ++i) {
            inverse[order[i]] = int32_t(i);
            xs[i] = order[i] % int32_t(gw) + 1;
            ys[i] = order[i] / int32_t(gw) + 1;
        }
        ov::Tensor mask(ov::element::f32, {1, 1, n, n});
        auto* additive = mask.data<float>();
        std::fill_n(additive, n * n, -std::numeric_limits<float>::infinity());
        size_t start = 0;
        for (size_t size : window_sizes) {
            for (size_t q = start; q < start + size; ++q)
                std::fill_n(additive + q * n + start, size, 0.f);
            start += size;
        }
        encoder.set_tensor("patch_indices", index_tensor(order));
        encoder.set_tensor("output_indices", index_tensor(inverse));
        encoder.set_tensor("position_x", index_tensor(xs));
        encoder.set_tensor("position_y", index_tensor(ys));
        encoder.set_tensor("merge_indices", index_tensor(merge_order(gh, gw, config.merge_size)));
        encoder.set_tensor("attention_mask", mask);
        encoder.set_tensor("pixel_values", pixels);
    }

private:
    size_t m_window;
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
        return std::make_shared<GGUFMuseVisionEncoder>(models.vision, models.processor, device, properties);
    case VLMModelType::QWEN3_5:
    case VLMModelType::QWEN3_5_MOE:
        return std::make_shared<GGUFQwenVisionEncoder>(models.vision, models.processor, device, properties);
    default:
        OPENVINO_THROW("Unsupported GGUF vision encoder family");
    }
}
}  // namespace ov::genai
