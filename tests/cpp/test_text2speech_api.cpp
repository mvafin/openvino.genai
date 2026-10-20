// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>
#include <utility>

#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"

namespace {
using Pipeline = ov::genai::Text2SpeechPipeline;
using Results = ov::genai::Text2SpeechDecodedResults;

static_assert(
    std::is_same_v<decltype(std::declval<Pipeline&>()
                                .generate(std::declval<const std::string&>(), std::declval<const ov::Tensor&>(), {})),
                   Results>);
static_assert(
    std::is_same_v<decltype(std::declval<Pipeline&>().generate(std::declval<const std::vector<std::string>&>(),
                                                               std::declval<const ov::Tensor&>(),
                                                               {})),
                   Results>);
static_assert(std::is_same_v<decltype(std::declval<Pipeline&>().generate_with_config(
                                 std::declval<const std::vector<std::string>&>(),
                                 std::declval<const ov::Tensor&>(),
                                 std::declval<const ov::genai::SpeechGenerationConfig&>())),
                             Results>);
}  // namespace
