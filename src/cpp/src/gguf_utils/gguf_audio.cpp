// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>

#include "gguf_multimodal.hpp"
#include "openvino/opsets/opset15.hpp"
#include "utils.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {
namespace {
// Gemma4 uses a magnitude HTK spectrogram, a 20 ms periodic Hann window,
// and semicausal padding. Whisper's power/Slaney frontend is not interchangeable.
std::shared_ptr<ov::Model> gemma4_spectrogram(size_t mel_bins) {
    using namespace ov::opset15;
    auto samples = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1});
    std::vector<float> window(512, 0.f);
    constexpr double pi = 3.14159265358979323846;
    for (size_t i = 0; i < 320; ++i)
        window[i] = .5f - .5f * std::cos(float(2 * pi * i / 320));
    auto spectrum = std::make_shared<STFT>(samples,
                                           Constant::create(ov::element::f32, {512}, window),
                                           Constant::create(ov::element::i64, {}, {512}),
                                           Constant::create(ov::element::i64, {}, {160}),
                                           false);
    auto power = std::make_shared<Multiply>(spectrum, spectrum);
    auto magnitude = std::make_shared<Sqrt>(
        std::make_shared<ReduceSum>(power, Constant::create(ov::element::i64, {1}, {-1}), false));
    std::vector<double> edges(mel_bins + 2);
    const double mel_max = 2595. * std::log10(1. + 8000. / 700.);
    for (size_t i = 0; i < edges.size(); ++i)
        edges[i] = 700. * (std::pow(10., mel_max * double(i) / double(mel_bins + 1) / 2595.) - 1.);
    std::vector<float> filters(mel_bins * 257);
    for (size_t m = 0; m < mel_bins; ++m)
        for (size_t k = 0; k < 257; ++k) {
            const double hz = double(k) * 16000. / 512.;
            filters[m * 257 + k] = std::max(0.,
                                            std::min((hz - edges[m]) / (edges[m + 1] - edges[m]),
                                                     (edges[m + 2] - hz) / (edges[m + 2] - edges[m + 1])));
        }
    auto mel =
        std::make_shared<MatMul>(Constant::create(ov::element::f32, {mel_bins, 257}, filters), magnitude, false, true);
    auto logged =
        std::make_shared<Log>(std::make_shared<Maximum>(mel, Constant::create(ov::element::f32, {}, {.001f})));
    auto output = std::make_shared<Unsqueeze>(logged, Constant::create(ov::element::i64, {2}, {0, 1}));
    return std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{samples});
}

class GGUFGemmaAudio {
public:
    GGUFGemmaAudio(const std::shared_ptr<ov::Model>& model, const std::string& device, const ov::AnyMap& properties) {
        const auto projector = model->get_rt_info<std::string>({"gguf_mmproj", "audio.projector"});
        m_unified = projector == "gemma4ua";
        OPENVINO_ASSERT(m_unified || projector == "gemma4a", "Unsupported GGUF audio projector: ", projector);
        const auto config = utils::get_model_properties(properties, "audio_encoder", device);
        m_encoder = utils::singleton_core().compile_model(model, device, config).create_infer_request();
        if (!m_unified) {
            const auto bins = std::stoull(model->get_rt_info<std::string>({"gguf_mmproj", "clip.audio.num_mel_bins"}));
            m_width = std::stoull(model->get_rt_info<std::string>({"gguf_mmproj", "clip.audio.embedding_length"}));
            m_features = utils::singleton_core()
                             .compile_model(gemma4_spectrogram(bins),
                                            "CPU",
                                            {{ov::hint::inference_precision.name(), ov::element::f32}})
                             .create_infer_request();
        }
    }

    std::vector<ov::Tensor> encode(const ov::Tensor& waveform) {
        std::lock_guard<std::mutex> lock(m_mutex);
        OPENVINO_ASSERT(waveform.get_element_type() == ov::element::f32 && waveform.get_size() > 0,
                        "Gemma4 audio requires nonempty mono f32 samples at 16000 Hz");
        const auto& shape = waveform.get_shape();
        OPENVINO_ASSERT(shape.size() == 1 || (shape.size() == 2 && shape[0] == 1), "Audio must be mono");
        const auto* samples = waveform.data<const float>();
        std::vector<ov::Tensor> outputs;
        const auto chunk_size = m_unified ? waveform.get_size() : size_t(30 * 16000);
        for (size_t off = 0; off < waveform.get_size(); off += chunk_size) {
            const auto count = std::min(chunk_size, waveform.get_size() - off);
            if (m_unified) {
                const auto frames = (count + 639) / 640;
                ov::Tensor input(ov::element::f32, {1, 1, frames, 640});
                std::fill_n(input.data<float>(), input.get_size(), 0.f);
                std::copy_n(samples + off, count, input.data<float>());
                m_encoder.set_tensor("waveform_frames", input);
            } else {
                const int64_t frames = (int64_t(count) + 160 - 321) / 160 + 1;
                OPENVINO_ASSERT(frames > 0, "Gemma4 audio chunk is too short to form a spectrogram frame");
                const auto padded = std::max(size_t((frames - 1) * 160 + 512), count + 160);
                ov::Tensor input(ov::element::f32, {padded});
                std::fill_n(input.data<float>(), input.get_size(), 0.f);
                std::copy_n(samples + off, count, input.data<float>() + 160);
                m_features.set_input_tensor(input);
                m_features.infer();
                m_encoder.set_tensor("features", m_features.get_output_tensor());
                set_positions((frames + 3) / 4);
            }
            m_encoder.infer();
            const auto output = m_encoder.get_tensor("audio_features");
            ov::Tensor owned(output.get_element_type(), output.get_shape());
            output.copy_to(owned);
            outputs.push_back(std::move(owned));
        }
        return outputs;
    }

private:
    void set_positions(size_t count) {
        ov::Tensor positions(ov::element::f32, {1, 1, 13, m_width});
        const auto half = m_width / 2;
        for (size_t p = 0; p < 13; ++p)
            for (size_t i = 0; i < half; ++i) {
                const float theta =
                    float(12 - p) * std::exp(-float(i) * (std::log(10000.f) / float(std::max(half - 1, size_t(1)))));
                positions.data<float>()[p * m_width + i] = std::sin(theta);
                positions.data<float>()[p * m_width + half + i] = std::cos(theta);
            }
        ov::Tensor mask(ov::element::f32, {1, 1, count, count});
        ov::Tensor relative(ov::element::i32, mask.get_shape());
        for (size_t q = 0; q < count; ++q)
            for (size_t k = 0; k < count; ++k) {
                const auto distance = int64_t(q) - int64_t(k);
                mask.data<float>()[q * count + k] = distance >= 0 && distance < 12 ? 0.f : -1e9f;
                relative.data<int32_t>()[q * count + k] = std::clamp(int64_t(12) - distance, int64_t(0), int64_t(12));
            }
        m_encoder.set_tensor("position_embeddings", positions);
        m_encoder.set_tensor("attention_mask", mask);
        m_encoder.set_tensor("relative_indices", relative);
    }

    bool m_unified = false;
    size_t m_width = 0;
    ov::InferRequest m_encoder, m_features;
    std::mutex m_mutex;
};
}  // namespace

void attach_gguf_audio_encoder(InputsEmbedder& embedder,
                               const GGUFMultimodalModels& models,
                               const std::string& device,
                               const ov::AnyMap& properties) {
    if (models.audio) {
        auto encoder = std::make_shared<GGUFGemmaAudio>(models.audio, device, properties);
        embedder.set_audio_encoder([encoder](const ov::Tensor& audio) {
            return encoder->encode(audio);
        });
    }
}
}  // namespace ov::genai
