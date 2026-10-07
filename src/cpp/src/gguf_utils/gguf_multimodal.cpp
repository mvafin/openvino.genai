// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gguf_multimodal.hpp"

#include <algorithm>
#include <nlohmann/json.hpp>
#include <sstream>

#include "gguf_tokenizer.hpp"
#include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#include "openvino/frontend/gguf/extension/genai.hpp"
#include "openvino/frontend/gguf/frontend.hpp"
#include "openvino/frontend/gguf/genai_vision.hpp"

namespace ov::genai {
namespace {
using nlohmann::json;

class MmprojMetadata {
public:
    explicit MmprojMetadata(const std::shared_ptr<ov::Model>& mmproj) : m_mmproj(mmproj) {}
    bool has(const std::string& key) const {
        return m_mmproj->has_rt_info({"gguf_mmproj", key});
    }
    std::string text(const std::string& key) const {
        return m_mmproj->get_rt_info<std::string>({"gguf_mmproj", key});
    }
    size_t integer(const std::string& key) const {
        return std::stoull(text(key));
    }
    size_t integer(const std::string& key, size_t fallback) const {
        return has(key) ? integer(key) : fallback;
    }
    std::vector<float> floats(const std::string& key) const {
        std::vector<float> values;
        std::istringstream stream(text(key));
        for (std::string field; std::getline(stream, field, ',');)
            values.push_back(std::stof(field));
        return values;
    }

private:
    std::shared_ptr<ov::Model> m_mmproj;
};

struct Configs {
    json config, image, video;
};

// GGUF stores only the encoder geometry; other settings are the HF processor defaults.
Configs hf_configs(const std::string& architecture,
                   const std::string& projector,
                   const MmprojMetadata& meta,
                   const GGUFMultimodalModels& models) {
    Configs c;
    const auto patch = meta.integer("clip.vision.patch_size");
    c.config["hidden_size"] = models.language->input("inputs_embeds").get_partial_shape()[2].get_length();
    c.config["vision_config"]["patch_size"] = patch;
    c.image["patch_size"] = patch;
    c.image["image_mean"] = meta.floats("clip.vision.image_mean");
    c.image["image_std"] = meta.floats("clip.vision.image_std");
    if (architecture == "gemma3") {
        const auto size = meta.integer("clip.vision.image_size");
        c.config["model_type"] = "gemma3";
        c.image["size"] = {{"height", size}, {"width", size}};
        c.video = c.image;
        return c;
    }
    if (architecture == "gemma4") {
        const bool unified = projector == "gemma4uv";
        c.config["model_type"] = unified ? "gemma4_unified" : "gemma4";
        if (const auto per_layer = models.vlm.find("text_embeddings_per_layer"))
            c.config["text_config"]["hidden_size_per_layer_input"] =
                per_layer->output().get_partial_shape()[3].get_length();
        const auto& inputs = models.language->inputs();
        if (std::any_of(inputs.begin(), inputs.end(), [](const ov::Output<ov::Node>& input) {
                return input.get_names().count("token_type_ids") > 0;
            }))
            c.config["text_config"]["use_bidirectional_attention"] = "vision";
        c.image["pooling_kernel_size"] = meta.integer("clip.vision.projector.scale_factor", 3);
        c.image["max_soft_tokens"] = unified ? 70 : 280;
        c.video = c.image;
        c.video["max_soft_tokens"] = 70;
        c.video["num_frames"] = 32;
        c.video["do_sample_frames"] = true;
        return c;
    }
    const auto merge = meta.integer("vision.merge");
    c.image["merge_size"] = merge;
    if (architecture == "muse-glimmer") {
        c.config["model_type"] = "muse_glimmer";
        // GGUF collapses HF's two-frame patch kernel.
        c.image["temporal_patch_size"] = 1;
        c.image["max_image_tokens"] = 4096;
        c.video = c.image;
        c.video["fps"] = 2.f;
        c.video["num_frames"] = 96;
        c.video["max_video_frame_tokens"] = 144;
        c.video["do_sample_frames"] = true;
        return c;
    }
    c.config["model_type"] = architecture == "qwen35moe" ? "qwen3_5_moe" : "qwen3_5";
    const auto side = meta.integer("clip.vision.image_size") / patch;
    c.config["vision_config"]["num_position_embeddings"] = side * side;
    c.image["temporal_patch_size"] = 2;
    c.image["size"] = {{"shortest_edge", meta.integer("clip.vision.image_min_pixels", 65536)},
                       {"longest_edge", meta.integer("clip.vision.image_max_pixels", 16777216)}};
    c.video = c.image;
    c.video["size"] = {{"shortest_edge", 4096}, {"longest_edge", 25165824}};
    c.video["fps"] = 2.f;
    c.video["min_frames"] = 4;
    c.video["max_frames"] = 768;
    c.video["do_sample_frames"] = true;
    return c;
}
}  // namespace

GGUFMultimodalModels read_gguf_multimodal(const std::filesystem::path& language,
                                          const std::filesystem::path& mmproj,
                                          const ov::AnyMap& properties) {
    namespace gguf = ov::frontend::gguf;
    GGUFMultimodalModels result;
    gguf::FrontEnd frontend;
    auto genai = std::make_shared<gguf::GenAIExtension>(gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
    frontend.add_extension(genai);
    result.language = frontend.convert(frontend.load(language.string()));
    auto tokenizer_metadata = take_gguf_tokenizer_metadata(result.language);
    result.stop_token_ids = gguf_stop_token_ids(tokenizer_metadata);
    result.tokenizer = Tokenizer(GGUFTokenizerParameters(std::move(tokenizer_metadata)), properties);
    gguf::FrontEnd projector_frontend;
    auto combined = projector_frontend.convert(projector_frontend.load(mmproj.string()));
    const MmprojMetadata meta(combined);
    const auto architecture = result.language->get_rt_info<std::string>("gguf_architecture");
    const auto projector = meta.text("vision.projector");
    const bool gemma4 = architecture == "gemma4" && (projector == "gemma4v" || projector == "gemma4uv");
    OPENVINO_ASSERT((architecture == "gemma3" && projector == "gemma3") ||
                        ((architecture == "qwen35" || architecture == "qwen35moe") && projector == "qwen3vl_merger") ||
                        (architecture == "muse-glimmer" && projector == "muse-glimmer") || gemma4,
                    "Unsupported GGUF language/projector pair: ",
                    architecture,
                    " / ",
                    projector);

    if (gemma4 && meta.has("audio.projector")) {
        result.audio = combined->clone();
        gguf::pass::AdaptMmprojToGenAI(gguf::pass::AdaptMmprojToGenAI::Modality::AUDIO).run_on_model(result.audio);
    }
    auto& vlm = result.vlm;
    vlm.models = gguf::genai_vision_models(combined);
    vlm.models["text_embeddings"] = genai->get_embedding_model();
    if (const auto& per_layer = genai->get_per_layer_embedding_model())
        vlm.models["text_embeddings_per_layer"] = per_layer;

    const auto configs = hf_configs(architecture, projector, meta, result);
    vlm.config = VLMConfig(configs.config);
    vlm.processor_config = ProcessorConfig(configs.image);
    vlm.video_processor_config = VideoProcessorConfig(configs.video);
    if (architecture == "gemma3") {
        // GGUF Gemma3 vocabularies lack <image_soft_token>; <pad> only marks image positions.
        vlm.config.image_soft_token = "<pad>";
        const auto placeholder = result.tokenizer.encode(vlm.config.image_soft_token, add_special_tokens(false));
        OPENVINO_ASSERT(placeholder.input_ids.get_size() == 1,
                        "GGUF Gemma3 requires a single-token padding placeholder for image assembly");
    }
    return result;
}
}  // namespace ov::genai
