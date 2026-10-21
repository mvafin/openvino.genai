// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/automatic_speech_recognition/pipeline.hpp"

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>

#include "automatic_speech_recognition/models/fun-asr/pipeline.hpp"
#include "automatic_speech_recognition/models/qwen3-asr/pipeline.hpp"
#include "automatic_speech_recognition/models/whisper/pipeline.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "utils.hpp"
#ifdef ENABLE_GGUF
#    include "gguf_utils/gguf_frontend_extensions.hpp"
#    include "gguf_utils/gguf_tokenizer.hpp"
#    include "gguf_utils/gguf_utils.hpp"
#    include "openvino/frontend/gguf/adapt_mmproj_to_genai.hpp"
#endif

namespace {

enum class ASRModelType { whisper, qwen3_asr, fun_asr };

ASRModelType read_model_type(const std::filesystem::path& models_path) {
    auto config_path = models_path / "config.json";
    std::ifstream stream(config_path);
    OPENVINO_ASSERT(stream.is_open(), "Failed to open '", config_path, "'");
    nlohmann::json parsed = nlohmann::json::parse(stream);
    OPENVINO_ASSERT(parsed.contains("model_type"),
                    "config.json must contain 'model_type' field for ASRPipeline model dispatch");
    std::string value = parsed.at("model_type").get<std::string>();

    static const std::unordered_map<std::string, ASRModelType> model_types_map = {
        {"whisper", ASRModelType::whisper},
        {"qwen3_asr", ASRModelType::qwen3_asr},
        {"fun_asr", ASRModelType::fun_asr},
    };

    auto it = model_types_map.find(value);
    if (it != model_types_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Unsupported '", value, "' ASR model type");
}

std::optional<ov::genai::ASRGenerationConfig> get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count(ov::genai::utils::CONFIG_ARG_NAME)) {
        return config_map.at(ov::genai::utils::CONFIG_ARG_NAME).as<ov::genai::ASRGenerationConfig>();
    }
    return std::nullopt;
}

}  // namespace

namespace ov::genai {

ASRPipeline::ASRPipeline(const std::filesystem::path& models_path,
                         const std::string& device,
                         const ov::AnyMap& properties) {
    const auto start_time = std::chrono::steady_clock::now();
#ifdef ENABLE_GGUF
    if (is_gguf_model(models_path)) {
        auto runtime = properties;
        const auto extensions = utils::extract_extensions_to_core(runtime);
        const auto projector = runtime.find(mmproj_path.name());
        OPENVINO_ASSERT(projector != runtime.end(), "GGUF ASR requires mmproj_path");
        const auto projector_path = projector->second.as<std::string>();
        runtime.erase(projector);
        auto gguf_properties = gguf_utils::extract_gguf_properties(runtime);
        OPENVINO_ASSERT(!gguf_properties.legacy_reader.value_or(false), "GGUF ASR requires gguf_reader=FRONTEND");
        runtime = std::move(gguf_properties.rest);
        namespace gguf = ov::frontend::gguf;
        gguf::FrontEnd frontend;
        const auto extension =
            register_gguf_extensions(frontend, extensions, gguf::GenAIExtension::InputMode::EMBEDS_TO_LOGITS);
        auto decoder = frontend.convert(frontend.load(models_path.string()));
        OPENVINO_ASSERT(decoder->get_rt_info<std::string>("gguf_architecture") == "qwen3vl",
                        "GGUF ASR requires a Qwen3-ASR decoder");
        auto tokenizer = GGUFTokenizer(GGUFTokenizerParameters(take_gguf_tokenizer_metadata(decoder)), runtime);
        gguf::FrontEnd projector_frontend;
        register_gguf_extensions(projector_frontend, extensions);
        auto encoder = projector_frontend.convert(projector_frontend.load(projector_path));
        OPENVINO_ASSERT(encoder->get_rt_info<std::string>({"gguf_mmproj", "audio.projector"}) == "qwen3a",
                        "GGUF ASR requires a qwen3a audio projector");
        const auto mel_bins =
            std::stoull(encoder->get_rt_info<std::string>({"gguf_mmproj", "clip.audio.num_mel_bins"}));
        gguf::pass::AdaptMmprojToGenAI(gguf::pass::AdaptMmprojToGenAI::Modality::AUDIO).run_on_model(encoder);
        m_impl = std::make_unique<Qwen3ASR>(tokenizer,
                                            encoder,
                                            decoder,
                                            extension->get_embedding_model(),
                                            mel_bins,
                                            device,
                                            runtime);
        m_impl->m_load_time_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time)
                .count();
        return;
    }
#endif
    const ASRModelType model_type = read_model_type(models_path);

    OPENVINO_ASSERT(!properties.count(ov::genai::forced_aligner.name()) || model_type == ASRModelType::qwen3_asr,
                    "'forced_aligner' is only supported for Qwen3-ASR models.");

    switch (model_type) {
    case ASRModelType::whisper: {
        m_impl = std::make_unique<WhisperASRPipelineAdapter>(models_path, device, properties);
        break;
    }
    case ASRModelType::qwen3_asr: {
        m_impl = std::make_unique<Qwen3ASR>(models_path, device, properties);
        break;
    }
    case ASRModelType::fun_asr: {
        m_impl = std::make_unique<FunASR>(models_path, device, properties);
        break;
    }
    }
    const auto stop_time = std::chrono::steady_clock::now();
    m_impl->m_load_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}

ASRDecodedResults ASRPipeline::generate(const AudioInputs& audio_inputs,
                                        const std::optional<ASRGenerationConfig>& generation_config,
                                        StreamerVariant streamer) {
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer, m_impl->m_tokenizer);
    return m_impl->generate(audio_inputs, generation_config, base_streamer);
}

ASRDecodedResults ASRPipeline::generate(const AudioInputs& audio_inputs, const ov::AnyMap& config_map) {
    auto config_arg = get_config_from_map(config_map);
    ASRGenerationConfig config = config_arg.has_value() ? *config_arg : get_generation_config();
    config.update_generation_config(config_map);

    StreamerVariant streamer_variant = utils::get_streamer_from_map(config_map);
    const std::shared_ptr<StreamerBase> base_streamer = utils::create_streamer(streamer_variant, m_impl->m_tokenizer);

    return m_impl->generate(audio_inputs, config, base_streamer);
}

Tokenizer ASRPipeline::get_tokenizer() {
    return m_impl->m_tokenizer;
}

ASRGenerationConfig ASRPipeline::get_generation_config() const {
    return m_impl->m_generation_config;
}

void ASRPipeline::set_generation_config(const ASRGenerationConfig& config) {
    m_impl->set_generation_config(config);
}

ASRPipeline::~ASRPipeline() = default;

}  // namespace ov::genai
