// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "gguf_modeling.hpp"
#include "gguf_tokenizer.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/adapt_to_genai.hpp"

namespace ov::genai {
GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties) {
    namespace gguf = ov::frontend::gguf;
    GGUFMultimodalModels result;
    result.language = convert_gguf_with_frontend(language.string());
    result.tokenizer = Tokenizer(GGUFTokenizerParameters(take_gguf_tokenizer_metadata(result.language)), properties);
    auto combined = convert_gguf_with_frontend(mmproj.string());
    const auto architecture = result.language->get_rt_info<std::string>("gguf_architecture");
    const auto projector = combined->get_rt_info<std::string>({"gguf_mmproj", "vision.projector"});
    const bool qwen = architecture == "qwen35" || architecture == "qwen35moe";
    const bool gemma4 = architecture == "gemma4" && (projector == "gemma4v" || projector == "gemma4uv");
    const bool muse = architecture == "muse-glimmer" && projector == "muse-glimmer";
    OPENVINO_ASSERT((architecture == "gemma3" && projector == "gemma3") || (qwen && projector == "qwen3vl_merger") ||
                        gemma4 || muse,
                    "Unsupported GGUF language/projector pair: ",
                    architecture,
                    " / ",
                    projector);

    // Read the processor metadata off the freshly converted mmproj before the adapter pass
    // rewrites the graph, so the model can then be adapted in place instead of cloned.
    const auto integer = [&](const std::string& key) {
        return std::stoull(combined->get_rt_info<std::string>({"gguf_mmproj", key}));
    };
    result.processor.size_height = result.processor.size_width = integer("clip.vision.image_size");
    result.processor.patch_size = integer("clip.vision.patch_size");
    result.config.vision_config_patch_size = result.processor.patch_size;
    const auto array = [&](const std::string& key, std::array<float, 3>& values, bool positive) {
        std::istringstream stream(combined->get_rt_info<std::string>({"gguf_mmproj", key}));
        for (size_t i = 0; i < values.size(); ++i) {
            std::string field;
            OPENVINO_ASSERT(std::getline(stream, field, ','), "Missing GGUF processor value ", key);
            values[i] = std::stof(field);
            OPENVINO_ASSERT(std::isfinite(values[i]) && (!positive || values[i] > 0),
                            "Invalid GGUF processor value ",
                            key);
        }
    };
    array("clip.vision.image_mean", result.processor.image_mean, /*positive=*/false);
    array("clip.vision.image_std", result.processor.image_std, /*positive=*/true);

    gguf::pass::AdaptToGenAI adapt(gguf::pass::AdaptToGenAI::InputMode::EMBEDS_TO_LOGITS);
    adapt.run_on_model(result.language);
    result.text_embeddings = adapt.get_embedding_model();
    for (const auto& output : result.text_embeddings->outputs()) {
        if (output.get_names().count("per_layer_inputs"))
            result.config.hidden_size_per_layer_input = output.get_partial_shape()[3].get_length();
    }
    if (gemma4 && combined->has_rt_info({"gguf_mmproj", "audio.projector"})) {
        result.audio = combined->clone();
        gguf::pass::AdaptMmprojToGenAI(gguf::pass::AdaptMmprojToGenAI::Modality::Audio).run_on_model(result.audio);
    }
    result.vision = combined;
    gguf::pass::AdaptMmprojToGenAI(gguf::pass::AdaptMmprojToGenAI::Modality::Vision).run_on_model(result.vision);
    const auto width = result.language->input("inputs_embeds").get_partial_shape()[2];
    OPENVINO_ASSERT(width == result.vision->output().get_partial_shape()[2],
                    "GGUF language and mmproj embedding widths do not match");
    result.config.hidden_size = width.get_length();
    result.config.scale_emb = 1.f;
    const auto set_pixel_bounds = [&](size_t min_tokens, size_t max_tokens) {
        const auto factor = result.processor.patch_size * result.processor.merge_size;
        result.processor.min_pixels = min_tokens * factor * factor;
        result.processor.max_pixels = max_tokens * factor * factor;
    };
    if (muse) {
        result.config.model_type = VLMModelType::MUSE_GLIMMER;
        result.processor.merge_size = integer("vision.merge");
        // llama.cpp's Muse GGUF stores the temporally collapsed image patch kernel.
        result.processor.temporal_patch_size = 1;
        result.processor.max_image_tokens = 4096;
        return result;
    }
    if (qwen) {
        result.config.model_type = architecture == "qwen35moe" ? VLMModelType::QWEN3_5_MOE : VLMModelType::QWEN3_5;
        result.processor.merge_size = integer("vision.merge");
        result.processor.temporal_patch_size = 2;
        set_pixel_bounds(8, 4096);
        for (auto entry : {std::make_pair("clip.vision.image_min_pixels", &result.processor.min_pixels),
                           std::make_pair("clip.vision.image_max_pixels", &result.processor.max_pixels)}) {
            if (combined->has_rt_info({"gguf_mmproj", entry.first}))
                *entry.second = integer(entry.first);
        }
        return result;
    }
    if (gemma4) {
        result.config.model_type = projector == "gemma4uv" ? VLMModelType::GEMMA4_UNIFIED : VLMModelType::GEMMA4;
        result.processor.merge_size = integer("vision.merge");
        if (projector == "gemma4uv") {
            result.processor.patch_size *= combined->has_rt_info({"gguf_mmproj", "clip.vision.projector.scale_factor"})
                                               ? integer("clip.vision.projector.scale_factor")
                                               : 3;
            result.processor.merge_size = 1;
        }
        result.config.vision_config_patch_size = result.processor.patch_size;
        set_pixel_bounds(70, 1120);
        // AdaptToGenAI adds token_type_ids exactly for models with bidirectional image attention.
        const auto& inputs = result.language->inputs();
        const bool token_types = std::any_of(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& input) {
            return input.get_names().count("token_type_ids") > 0;
        });
        result.config.use_bidirectional_attention = token_types ? "vision" : "";
        return result;
    }
    result.config.model_type = VLMModelType::GEMMA3;
    // GGUF Gemma3 vocabularies do not contain HF's added <image_soft_token>.
    // Use the existing padding token only as an assembly placeholder; every occurrence
    // is replaced with a projected image vector before language-model inference.
    result.config.image_soft_token = "<pad>";
    const auto placeholder = result.tokenizer.encode(result.config.image_soft_token, add_special_tokens(false));
    OPENVINO_ASSERT(placeholder.input_ids.get_size() == 1,
                    "GGUF Gemma3 requires a single-token padding placeholder for image assembly");
    return result;
}
}  // namespace ov::genai
