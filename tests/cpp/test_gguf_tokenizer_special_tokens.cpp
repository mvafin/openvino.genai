// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

#include "gguf_utils/gguf_lfm2_audio.hpp"
#include "gguf_utils/gguf_multimodal.hpp"
#include "gtest/gtest.h"
#include "openvino/genai/gguf.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/opsets/opset15.hpp"
#include "visual_language/clip.hpp"
#include "visual_language/embedding_model.hpp"
#include "visual_language/gemma3/classes.hpp"
#include "visual_language/gemma3n/classes.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/lfm2/classes.hpp"
#include "visual_language/minicpmv4_7/classes.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/qwen3_omni/classes.hpp"
#include "visual_language/vlm_chat_context.hpp"
#include "visual_language/vlm_utils.hpp"

namespace {
ov::genai::GGUFTokenizerParameters sentencepiece_config() {
    std::vector<std::string> vocab{"<unk>",
                                   "<bos>",
                                   "<eos>",
                                   "<pad>",
                                   "<start_of_turn>",
                                   "<end_of_turn>",
                                   "<start_of_image>",
                                   "<end_of_image>",
                                   "▁",
                                   "a",
                                   "b"};
    std::vector<int32_t> types{2, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1};
    for (const auto* token : {"<|audio|>", "<|audio>", "<audio|>", "<|image|>", "<|video|>", "<|audio_start|>"}) {
        vocab.emplace_back(token);
        types.push_back(3);
    }
    for (int i = 0; i < 256; ++i) {
        char byte[7];
        std::snprintf(byte, sizeof(byte), "<0x%02X>", i);
        vocab.emplace_back(byte);
        types.push_back(6);
    }
    ov::Tensor type_tensor(ov::element::i32, {types.size()});
    std::memcpy(type_tensor.data(), types.data(), type_tensor.get_byte_size());
    ov::Tensor scores(ov::element::f32, {types.size()});
    std::fill_n(scores.data<float>(), scores.get_size(), -1.f);
    const auto id = [](uint32_t value) {
        ov::Tensor tensor(ov::element::u32, {});
        tensor.data<uint32_t>()[0] = value;
        return tensor;
    };
    ov::Tensor no_prefix(ov::element::boolean, {}), add_bos(ov::element::boolean, {});
    no_prefix.data<bool>()[0] = false;
    add_bos.data<bool>()[0] = true;
    return ov::genai::GGUFTokenizerParameters(
        {{"model", std::string("llama")},
         {"tokens", vocab},
         {"token_type", type_tensor},
         {"scores", scores},
         {"unknown_token_id", id(0)},
         {"bos_token_id", id(1)},
         {"eos_token_id", id(2)},
         {"padding_token_id", id(3)},
         {"add_space_prefix", no_prefix},
         {"add_bos_token", add_bos},
         {"chat_template", std::string("{{ bos_token }}<start_of_turn>{{ messages[0]['content'] }}<end_of_turn>")}});
}
constexpr size_t vocab_size = 273;  // tokens in sentencepiece_config()

ov::genai::GGUFTokenizerParameters combining_marks_config(const std::string& pre) {
    ov::Tensor types(ov::element::i32, {9});
    std::fill_n(types.data<int32_t>(), 6, 1);
    types.data<int32_t>()[6] = 2;
    types.data<int32_t>()[7] = types.data<int32_t>()[8] = 3;
    ov::Tensor unknown_id(ov::element::u32, {});
    unknown_id.data<uint32_t>()[0] = 6;
    ov::Tensor bos_id(ov::element::u32, {}), eos_id(ov::element::u32, {});
    bos_id.data<uint32_t>()[0] = 7;
    eos_id.data<uint32_t>()[0] = 8;
    ov::Tensor add_bos(ov::element::boolean, {});
    add_bos.data<bool>()[0] = false;
    return ov::genai::GGUFTokenizerParameters(
        {{"model", std::string("gpt2")},
         {"pre", pre},
         {"tokens", std::vector<std::string>{"a", "Ì", "ģ", "Ìģ", "aÌ", "aÌģ", "<unk>", "<bos>", "<eos>"}},
         {"merges", std::vector<std::string>{"Ì ģ", "a Ì", "aÌ ģ", "a Ìģ"}},
         {"token_type", types},
         {"unknown_token_id", unknown_id},
         {"bos_token_id", bos_id},
         {"eos_token_id", eos_id},
         {"add_bos_token", add_bos}});
}

// Text embeddings are all 0.25; the vision encoder is never run.
ov::genai::VLMModels gemma4_models() {
    ov::genai::VLMModels models;
    models.config.model_type = ov::genai::VLMModelType::GEMMA4;
    models.config.hidden_size = 4;
    models.config.video_token = "<|video|>";
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto table = ov::op::v0::Constant::create(ov::element::f32, {vocab_size, 4}, {0.25f});
    auto lookup =
        std::make_shared<ov::op::v8::Gather>(table, ids, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    models.models["text_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{lookup}, ov::ParameterVector{ids});
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 12});
    pixels->output(0).set_names({"pixel_values"});
    models.models["vision_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{pixels}, ov::ParameterVector{pixels});
    return models;
}

ov::genai::VLMModels qwen3_split_models(ov::genai::VLMModelType type) {
    auto models = gemma4_models();
    models.config.model_type = type;
    models.config.vision_start_token = "<start_of_image>";
    models.config.vision_end_token = "<end_of_image>";
    models.config.image_pad_token = "<pad>";
    models.config.video_pad_token = "<|video|>";
    models.config.vision_config_num_position_embeddings = 4;
    models.processor_config.merge_size = 1;
    models.processor_config.preserve_native_resolution = true;
    auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{4, -1});
    indices->output(0).set_names({"input"});
    auto positions = std::make_shared<ov::op::v8::Gather>(ov::op::v0::Constant::create(ov::element::f32, {4, 4}, {1.f}),
                                                          indices,
                                                          ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    models.models["vision_embeddings_pos"] =
        std::make_shared<ov::Model>(ov::OutputVector{positions}, ov::ParameterVector{indices});
    auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    hidden->output(0).set_names({"hidden_states"});
    auto rotary = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    rotary->output(0).set_names({"rotary_pos_emb"});
    auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, -1});
    mask->output(0).set_names({"attention_mask"});
    auto output = std::make_shared<ov::op::v1::Add>(hidden, ov::op::v0::Constant::create(ov::element::f32, {}, {0.f}));
    output->output(0).set_names({"last_hidden_state"});
    models.models["vision_embeddings_merger"] =
        std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{hidden, rotary, mask});
    return models;
}

// Two audio tokens whose features all equal the first sample, so tests can trace their placement.
std::vector<ov::Tensor> first_sample_audio_features(const ov::Tensor& audio) {
    ov::Tensor features(ov::element::f32, {1, 2, 4});
    std::fill_n(features.data<float>(), features.get_size(), audio.data<const float>()[0]);
    return {features};
}
}  // namespace

TEST(GGUFTokenizer, SentencePieceControlTokensUseTheirVocabularyIds) {
    ov::genai::GGUFTokenizer tokenizer(sentencepiece_config());
    auto encoded = tokenizer.encode("<start_of_turn>a<end_of_turn><start_of_image><pad><end_of_image>",
                                    ov::genai::add_special_tokens(false));
    const std::vector<int64_t> expected{4, 9, 5, 6, 3, 7};
    ASSERT_EQ(encoded.input_ids.get_size(), expected.size());
    EXPECT_EQ(std::vector<int64_t>(encoded.input_ids.data<int64_t>(),
                                   encoded.input_ids.data<int64_t>() + encoded.input_ids.get_size()),
              expected);
    auto with_bos = tokenizer.encode("a", ov::genai::add_special_tokens(true));
    ASSERT_EQ(with_bos.input_ids.get_size(), 2);
    EXPECT_EQ(with_bos.input_ids.data<int64_t>()[0], 1);
    EXPECT_EQ(tokenizer.get_bos_token(), "<bos>");
    EXPECT_EQ(tokenizer.get_eos_token(), "<eos>");
    EXPECT_EQ(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, false),
              "<bos><start_of_turn>a<end_of_turn>");
}

TEST(GGUFTokenizer, ChatTemplateSupportsUndefinedAndAdjacentStringLiterals) {
    auto config = sentencepiece_config();
    config.config["chat_template"] =
        std::string("{% if enable_thinking is undefined %}{{ \"first\"\n  \"second\" }}{% endif %}"
                    "{{ messages[0]['content'] }}");
    ov::genai::GGUFTokenizer tokenizer(config);
    EXPECT_EQ(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, false), "firstseconda");
}

TEST(GGUFTokenizer, GPT4OSplitsCombiningMarksLikeLlamaCPU) {
    // Pinned llama.cpp 03fa73cb: "a\u0301" splits into "a" and the acute accent.
    const std::vector<int64_t> expected{0, 3};
    for (const auto& pre : {"gpt-4o", "llama4"}) {
        ov::genai::GGUFTokenizer tokenizer(combining_marks_config(pre));
        const auto ids = tokenizer.encode("a\u0301", ov::genai::add_special_tokens(false)).input_ids;
        ASSERT_EQ(ids.get_size(), expected.size()) << pre;
        EXPECT_EQ(std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size()), expected) << pre;
    }
}

TEST(GGUFTokenizer, Gemma4UsesMergesForOrdinaryVocabularyEntries) {
    auto config = combining_marks_config("gemma4");
    config.config["model"] = std::string("gemma4");
    config.config["tokens"] =
        std::vector<std::string>{"<unk>", "<bos>", "<eos>", "a", "b", "ab", "c", "abc", "\n", "\n\n"};
    config.config["merges"] = std::vector<std::string>{"a b"};
    ov::Tensor types(ov::element::i32, {10});
    std::fill_n(types.data<int32_t>(), 10, 1);
    types.data<int32_t>()[0] = 2;
    types.data<int32_t>()[1] = types.data<int32_t>()[2] = 3;
    config.config["token_type"] = types;
    for (const auto& [key, id] : std::vector<std::pair<std::string, uint32_t>>{{"unknown_token_id", 0},
                                                                               {"bos_token_id", 1},
                                                                               {"eos_token_id", 2}}) {
        ov::Tensor value(ov::element::u32, {});
        value.data<uint32_t>()[0] = id;
        config.config[key] = value;
    }
    ov::genai::GGUFTokenizer tokenizer(config);
    const auto ids = tokenizer.encode("abc\n\n", ov::genai::add_special_tokens(false)).input_ids;
    // Pinned llama.cpp 03fa73cb: "abc" needs a merge; repeated newlines use their vocabulary entry.
    const std::vector<int64_t> expected{5, 6, 9};
    ASSERT_EQ(ids.get_size(), expected.size());
    EXPECT_EQ(std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size()), expected);
}

TEST(GGUFTokenizer, Gemma4ByteTokenSpellingUsesOrdinaryCharacters) {
    auto config = combining_marks_config("gemma4");
    config.config["model"] = std::string("gemma4");
    config.config["tokens"] =
        std::vector<std::string>{"<unk>", "<bos>", "<eos>", "<", "0", "x", "8", "5", ">", "<0x85>", "\n", "<0x0A>"};
    config.config["merges"] = std::vector<std::string>{};
    ov::Tensor types(ov::element::i32, {12});
    std::fill_n(types.data<int32_t>(), 12, 1);
    types.data<int32_t>()[0] = 2;
    types.data<int32_t>()[1] = types.data<int32_t>()[2] = 3;
    types.data<int32_t>()[9] = 6;
    types.data<int32_t>()[11] = 6;
    config.config["token_type"] = types;
    for (const auto& [key, id] : std::vector<std::pair<std::string, uint32_t>>{{"unknown_token_id", 0},
                                                                               {"bos_token_id", 1},
                                                                               {"eos_token_id", 2}}) {
        ov::Tensor value(ov::element::u32, {});
        value.data<uint32_t>()[0] = id;
        config.config[key] = value;
    }
    ov::genai::GGUFTokenizer tokenizer(config);
    const auto ids = tokenizer.encode("<0x85>", ov::genai::add_special_tokens(false)).input_ids;
    const std::vector<int64_t> expected{3, 4, 5, 6, 7, 8};
    ASSERT_EQ(ids.get_size(), expected.size());
    EXPECT_EQ(std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size()), expected);
    const auto byte_ids = tokenizer.encode(std::string(1, char(0x85)), ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(byte_ids.get_size(), 1);
    EXPECT_EQ(byte_ids.data<int64_t>()[0], 9);
    const auto newline_ids = tokenizer.encode("\n", ov::genai::add_special_tokens(false)).input_ids;
    ASSERT_EQ(newline_ids.get_size(), 1);
    EXPECT_EQ(newline_ids.data<int64_t>()[0], 10);
    EXPECT_EQ(tokenizer.decode(std::vector<int64_t>{11}), "\n");
}

TEST(GGUFMultimodal, Qwen3VLAllowsMissingOptionalDeepStackOutput) {
    auto embeddings = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    embeddings->output(0).set_names({"inputs_embeds"});
    auto merger = std::make_shared<ov::Model>(ov::OutputVector{embeddings}, ov::ParameterVector{embeddings});
    EXPECT_TRUE(ov::genai::gguf_qwen3vl_deepstack_visual_indexes(merger).empty());
    EXPECT_THROW(ov::genai::gguf_qwen3vl_deepstack_visual_indexes(nullptr), ov::Exception);
}

TEST(GGUFMultimodal, Qwen3VLDeepStackIndexesFollowOutputLayerCount) {
    auto features = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{2, -1, 4});
    features->output(0).set_names({"deepstack_feature_lists", "features"});
    auto merger = std::make_shared<ov::Model>(ov::OutputVector{features}, ov::ParameterVector{features});
    EXPECT_EQ(ov::genai::gguf_qwen3vl_deepstack_visual_indexes(merger), (std::vector<size_t>{0, 1}));
}

TEST(GGUFMultimodal, Qwen3VLRejectsMalformedPresentDeepStackOutput) {
    for (const auto& shape : {ov::PartialShape{2, 4}, ov::PartialShape::dynamic(), ov::PartialShape{-1, -1, 4}}) {
        SCOPED_TRACE(shape);
        auto features = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
        features->output(0).set_names({"deepstack_feature_lists"});
        auto merger = std::make_shared<ov::Model>(ov::OutputVector{features}, ov::ParameterVector{features});
        EXPECT_THROW(ov::genai::gguf_qwen3vl_deepstack_visual_indexes(merger), ov::Exception);
    }
}

TEST(GGUFMultimodal, Qwen3SplitConsumersRunTextAndImagesWithoutDeepStack) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    for (const auto type : {VLMModelType::QWEN3_VL, VLMModelType::QWEN3_OMNI}) {
        auto models = qwen3_split_models(type);
        std::unique_ptr<InputsEmbedderQwen3VL> embedder;
        if (type == VLMModelType::QWEN3_OMNI)
            embedder = std::make_unique<InputsEmbedderQwen3Omni>(models, tokenizer, "CPU", ov::AnyMap{});
        else
            embedder = std::make_unique<InputsEmbedderQwen3VL>(models, tokenizer, "CPU", ov::AnyMap{});
        embedder->set_apply_chat_template_status(false);
        EXPECT_TRUE(embedder->get_lm_extra_inputs().empty());
        VLMPerfMetrics metrics;
        const auto text = embedder->get_inputs_embeds("a", {}, {}, metrics);
        ASSERT_EQ(text.get_shape(), (ov::Shape{1, 2, 4}));
        EXPECT_TRUE(embedder->get_lm_extra_inputs().empty());
        embedder->finish_chat();
        EncodedImage image{ov::Tensor(ov::element::f32, {4, 4})};
        image.resized_source_size = {2, 2};
        std::fill_n(image.resized_source.data<float>(), image.resized_source.get_size(), 7.f);
        const auto embeddings = embedder->get_inputs_embeds("<start_of_image><pad><pad><pad><pad><end_of_image>",
                                                            {image},
                                                            {},
                                                            metrics,
                                                            true,
                                                            {0});
        ASSERT_EQ(embeddings.get_shape(), (ov::Shape{1, 7, 4}));
        for (size_t token = 2; token < 6; ++token)
            for (size_t channel = 0; channel < 4; ++channel)
                EXPECT_FLOAT_EQ(embeddings.data<const float>()[token * 4 + channel], 8.f);
        EXPECT_TRUE(embedder->get_lm_extra_inputs().empty());
        EXPECT_EQ(image.resized_source.data<const float>()[0], 7.f);
    }
}

TEST(GGUFMultimodal, Qwen3SplitConsumersRetainConfiguredDeepStackInputs) {
    using namespace ov::genai;
    auto models = qwen3_split_models(VLMModelType::QWEN3_VL);
    models.config.vision_config_deepstack_visual_indexes = {0, 1};
    GGUFTokenizer tokenizer(sentencepiece_config());
    InputsEmbedderQwen3VL embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    VLMPerfMetrics metrics;
    embedder.get_inputs_embeds("a", {}, {}, metrics);
    const auto& extras = embedder.get_lm_extra_inputs();
    EXPECT_EQ(extras.size(), 2);
    EXPECT_EQ(extras.at("deepstack_visual_embeds").get_shape(), (ov::Shape{2, 1, 4}));
    ASSERT_EQ(extras.at("visual_pos_masks").get_shape(), (ov::Shape{1, 1}));
    EXPECT_FALSE(extras.at("visual_pos_masks").data<const bool>()[0]);
}

TEST(GGUFMultimodal, OmniSplitPositionsMatchDecoderSectionsInPrefillAndDecode) {
    using namespace ov::genai;
    struct TestEmbedder : InputsEmbedderQwen3Omni {
        using InputsEmbedderQwen3Omni::get_generation_phase_position_ids;
        using InputsEmbedderQwen3Omni::InputsEmbedderQwen3Omni;
        using InputsEmbedderQwen3VL::get_inputs_embeds;
    };
    GGUFTokenizer tokenizer(sentencepiece_config());
    const std::vector<int64_t> spatial{0, 1, 2, 2, 2, 2, 4, 0, 1, 2, 2, 3, 3, 4, 0, 1, 2, 3, 2, 3, 4};
    for (const size_t sections : {3, 4}) {
        auto models = qwen3_split_models(VLMModelType::QWEN3_OMNI);
        models.config.position_ids_sections = sections;
        TestEmbedder embedder(models, tokenizer, "CPU", {});
        embedder.set_apply_chat_template_status(false);
        VLMPerfMetrics metrics;
        embedder.get_inputs_embeds("a", {}, {}, metrics);
        const auto text_positions = embedder.get_position_ids(2, 0).first;
        ASSERT_EQ(text_positions.get_shape(), (ov::Shape{sections, 1, 2}));
        for (size_t section = 0; section < sections; ++section) {
            EXPECT_EQ(text_positions.data<const int64_t>()[section * 2], 0);
            EXPECT_EQ(text_positions.data<const int64_t>()[section * 2 + 1], 1);
        }
        embedder.finish_chat();
        EncodedImage image{ov::Tensor(ov::element::f32, {4, 4})};
        image.resized_source_size = {2, 2};
        std::fill_n(image.resized_source.data<float>(), image.resized_source.get_size(), 7.f);
        embedder
            .get_inputs_embeds("<start_of_image><pad><pad><pad><pad><end_of_image>", {image}, {}, metrics, true, {0});
        const auto [positions, rope_delta] = embedder.get_position_ids(7, 0);
        ASSERT_EQ(positions.get_shape(), (ov::Shape{sections, 1, 7}));
        ASSERT_TRUE(rope_delta.has_value());
        EXPECT_EQ(*rope_delta, -2);
        EXPECT_EQ(std::vector<int64_t>(positions.data<const int64_t>(), positions.data<const int64_t>() + 21), spatial);
        if (sections == 4)
            EXPECT_EQ(std::vector<int64_t>(positions.data<const int64_t>() + 21, positions.data<const int64_t>() + 28),
                      (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 4}));
        const auto [decode, decode_delta] = embedder.get_generation_phase_position_ids(2, 7, *rope_delta);
        ASSERT_EQ(decode.get_shape(), (ov::Shape{sections, 1, 2}));
        EXPECT_EQ(decode_delta, rope_delta);
        for (size_t section = 0; section < sections; ++section) {
            EXPECT_EQ(decode.data<const int64_t>()[section * 2], 5);
            EXPECT_EQ(decode.data<const int64_t>()[section * 2 + 1], 6);
        }
        EXPECT_TRUE(embedder.get_lm_extra_inputs().empty());
    }
    auto invalid = qwen3_split_models(VLMModelType::QWEN3_OMNI);
    invalid.config.position_ids_sections = 2;
    EXPECT_THROW((TestEmbedder(invalid, tokenizer, "CPU", {})), ov::Exception);
}

TEST(GGUFMultimodal, PreformattedChatDoesNotDuplicateSpecialTokens) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    struct TestEmbedder : InputsEmbedderGemma4 {
        using InputsEmbedderGemma4::InputsEmbedderGemma4;
        using IInputsEmbedder::get_encoded_input_ids;
    };
    TestEmbedder embedder(gemma4_models(), tokenizer, "CPU", {});
    const auto encode = [&](const std::string& prompt) {
        VLMPerfMetrics metrics;
        auto ids = embedder.get_encoded_input_ids(prompt, metrics);
        return std::vector<int64_t>(ids.data<int64_t>(), ids.data<int64_t>() + ids.get_size());
    };
    const std::vector<int64_t> expected{1, 4, 9, 5};
    EXPECT_EQ(encode("a"), expected);
    embedder.finish_chat();
    embedder.set_apply_chat_template_status(false, true);
    EXPECT_EQ(encode(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, true)), expected);
    embedder.finish_chat();
    embedder.set_apply_chat_template_status(false);
    EXPECT_EQ(encode("a"), (std::vector<int64_t>{1, 9}));
    embedder.finish_chat();
    embedder.set_apply_chat_template_status(true);
    EXPECT_EQ(encode("a"), expected);
}

TEST(GGUFMultimodal, GemmaAudioPlacementFollowsTheAudioSequence) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    InputsEmbedderGemma4 embedder(gemma4_models(), tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder(first_sample_audio_features);
    const auto sample = [](float value) {
        ov::Tensor audio(ov::element::f32, {1});
        audio.data<float>()[0] = value;
        return audio;
    };
    const auto encoded = embedder.encode_audios({sample(7.f), ov::Tensor(ov::element::f32, {0}), sample(9.f)});
    ASSERT_EQ(encoded.size(), 3);
    EXPECT_EQ(encoded[0].num_audio_tokens, 2);
    EXPECT_EQ(encoded[0].audio_features.get_shape(), (ov::Shape{2, 4}));
    EXPECT_EQ(encoded[1].num_audio_tokens, 0);
    // The first value of each feature run, in prompt order.
    const auto placed = [&](const NormalizedPrompt& prompt, size_t base) {
        auto sequence = prompt.audios_sequence;
        vlm_utils::rebase_media_sequence(sequence, base);
        VLMPerfMetrics metrics;
        const auto features =
            embedder.get_inputs_embeds(prompt.unified_prompt, {}, {}, encoded, metrics, true, {}, {}, sequence, {});
        std::vector<float> runs;
        const auto* data = features.data<const float>();
        for (size_t i = 0; i < features.get_size(); i += 4)
            if (data[i] != 0.25f && (runs.empty() || runs.back() != data[i]))
                runs.push_back(data[i]);
        return runs;
    };
    const auto native = embedder.normalize_prompt("a<|audio|>b<|audio|><|audio|>", 0, 0, 0, {}, {}, encoded);
    EXPECT_EQ(native.audios_sequence, (std::vector<size_t>{0, 1, 2}));
    EXPECT_EQ(native.unified_prompt, "a<|audio><|audio|><|audio|><audio|>b<|audio><audio|><|audio><|audio|><|audio|><audio|>");
    EXPECT_EQ(placed(native, 0), (std::vector<float>{7.f, 9.f}));
    // A later chat turn with history audios 0-1: reversed universal tags bind by index.
    const auto universal = embedder.normalize_prompt("<ov_genai_audio_4>x<ov_genai_audio_2>", 0, 0, 2, {}, {}, encoded);
    EXPECT_EQ(universal.audios_sequence, (std::vector<size_t>{4, 2}));
    EXPECT_EQ(placed(universal, 2), (std::vector<float>{9.f, 7.f}));
    // Audio-free prompts are untouched, and a tag without its audio is rejected.
    EXPECT_EQ(embedder.normalize_prompt("ab", 0, 0, 0, {}, {}, {}).unified_prompt, "ab");
    EXPECT_ANY_THROW(embedder.normalize_prompt("<ov_genai_audio_0>", 0, 0, 0, {}, {}, {}));
    EXPECT_TRUE(embedder.encode_audios({}).empty());
}

TEST(GGUFMultimodal, GemmaPerLayerInputsUseTheirLookupModel) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    constexpr size_t layers = 2, width = 3;
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    // Row i of the per-layer table holds i, so each value identifies the looked-up token.
    std::vector<float> rows(vocab_size * layers * width);
    for (size_t i = 0; i < rows.size(); ++i)
        rows[i] = float(i / (layers * width));
    auto per_layer_table = ov::op::v0::Constant::create(ov::element::f32, {vocab_size, layers * width}, rows);
    auto per_layer_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    per_layer_ids->output(0).set_names({"input_ids"});
    auto per_layer = std::make_shared<ov::op::v1::Reshape>(
        std::make_shared<ov::op::v8::Gather>(per_layer_table, per_layer_ids, axis),
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, layers, width}),
        true);
    per_layer->output(0).set_names({"per_layer_inputs"});
    auto models = gemma4_models();
    models.config.hidden_size_per_layer_input = width;
    models.models["text_embeddings_per_layer"] =
        std::make_shared<ov::Model>(ov::OutputVector{per_layer}, ov::ParameterVector{per_layer_ids});
    InputsEmbedderGemma4 embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder([](const ov::Tensor&) {
        return std::vector<ov::Tensor>{ov::Tensor(ov::element::f32, {1, 2, 4})};
    });
    const auto audios = embedder.encode_audios({ov::Tensor(ov::element::f32, {1})});
    const auto normalized = embedder.normalize_prompt("a<|audio|>b", 0, 0, 0, {}, {}, audios);
    const auto& prompt = normalized.unified_prompt;
    VLMPerfMetrics metrics;
    const auto inputs_embeds =
        embedder.get_inputs_embeds(prompt, {}, {}, audios, metrics, true, {}, {}, normalized.audios_sequence, {});
    const auto token_ids = tokenizer.encode(prompt).input_ids;
    const auto& per_layer_inputs = embedder.get_lm_extra_inputs().at("per_layer_inputs");
    ASSERT_EQ(per_layer_inputs.get_shape(), (ov::Shape{1, inputs_embeds.get_shape()[1], layers, width}));
    ASSERT_EQ(token_ids.get_size(), inputs_embeds.get_shape()[1]);
    for (size_t t = 0; t < token_ids.get_size(); ++t) {
        for (size_t i = 0; i < layers * width; ++i)
            EXPECT_EQ(per_layer_inputs.data<const float>()[t * layers * width + i],
                      float(token_ids.data<const int64_t>()[t]));
    }
    // Generated tokens use the same lookup through the continuous-batching callback.
    ov::Tensor generated(ov::element::i64, {1, 1});
    generated.data<int64_t>()[0] = 9;
    const auto decoded = embedder.get_per_layer_embeddings_callback()(generated);
    ASSERT_EQ(decoded.get_shape(), (ov::Shape{1, 1, layers, width}));
    EXPECT_EQ(decoded.data<const float>()[0], 9.f);
}

TEST(GGUFMultimodal, ModernAudioHistorySwitchEditAndRollback) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    InputsEmbedder embedder(gemma4_models(), tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder(first_sample_audio_features);
    auto registry = std::make_shared<VisionRegistry>();
    const auto audio = [](float value) {
        ov::Tensor tensor(ov::element::f32, {1});
        tensor.data<float>()[0] = value;
        return tensor;
    };
    // Embeds the whole normalized history with the audios the chat context resolved for it.
    const auto count_features = [&](const VLMChatContext::ProcessedChatData& data, float value) {
        std::string prompt;
        for (size_t i = 0; i < data.normalized_history.size(); ++i)
            prompt += data.normalized_history[i]["content"].get_string();
        VLMPerfMetrics metrics;
        const auto features = embedder.get_inputs_embeds(
            prompt, {}, {}, data.encoded_audios, metrics, true, {}, {}, data.audio_sequence, {});
        return std::count(features.data<const float>(), features.data<const float>() + features.get_size(), value);
    };
    ChatHistory first({{{"role", "user"}, {"content", "a<|audio|>b"}}});
    auto first_data = VLMChatContext(first, registry, embedder).process({}, {}, {}, {audio(7.f)});
    EXPECT_EQ(count_features(first_data, 7.f), 8);
    ChatHistory second({{{"role", "user"}, {"content", "<|audio|>"}}});
    auto second_data = VLMChatContext(second, registry, embedder).process({}, {}, {}, {audio(9.f)});
    EXPECT_EQ(count_features(second_data, 9.f), 8);
    first.push_back({{"role", "assistant"}, {"content", "a"}});
    first.push_back({{"role", "user"}, {"content", "b"}});
    first_data = VLMChatContext(first, registry, embedder).process({});
    EXPECT_EQ(count_features(first_data, 7.f), 8);
    EXPECT_EQ(count_features(first_data, 9.f), 0);
    first.push_back({{"role", "user"}, {"content", "<|audio|>"}});
    VLMChatContext cancelled(first, registry, embedder);
    auto cancelled_data = cancelled.process({}, {}, {}, {audio(11.f)});
    EXPECT_EQ(count_features(cancelled_data, 11.f), 8);
    cancelled.rollback();
    first.pop_back();
    first_data = VLMChatContext(first, registry, embedder).process({});
    EXPECT_EQ(count_features(first_data, 7.f), 8);
    EXPECT_EQ(count_features(first_data, 11.f), 0);
    first[0]["content"] = "b<|audio|>a";
    first.pop_back();
    first.pop_back();
    first_data = VLMChatContext(first, registry, embedder).process({}, {}, {}, {audio(13.f)});
    EXPECT_EQ(count_features(first_data, 13.f), 8);
    EXPECT_EQ(count_features(first_data, 7.f), 0);
    first[0]["content"] = "a<|audio|>";
    VLMChatContext edited(first, registry, embedder);
    edited.process({}, {}, {}, {audio(17.f)});
    edited.rollback();
    EXPECT_TRUE(ChatHistoryInternalState::get_or_create(first, registry)->get_messages_metadata().empty());
}

TEST(GGUFMultimodal, ReusingTextEmbeddingsDoesNotAccumulateScaling) {
    using namespace ov::genai;
    auto model = gemma4_models().at("text_embeddings");
    const auto original_output = model->output().get_node_shared_ptr();
    ov::Tensor ids(ov::element::i64, {1, 1});
    ids.data<int64_t>()[0] = 0;
    for (float scale : {2.f, 3.f}) {
        EmbeddingsModel embeddings(model, scale, "CPU", {});
        CircularBufferQueueElementGuard<EmbeddingsRequest> guard(embeddings.get_request_queue().get());
        auto& request = guard.get();
        const auto output = embeddings.infer(request, ids);
        EXPECT_FLOAT_EQ(output.data<float>()[0], .25f * scale);
        EXPECT_EQ(model->output().get_node_shared_ptr(), original_output);
    }
}

TEST(GGUFMultimodal, Gemma3PositionOffsetAppliesDuringPrefillAndDecode) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::GEMMA3;
    models.config.image_soft_token = "<pad>";
    for (size_t offset : {0u, 1u}) {
        models.config.position_ids_offset = offset;
        InputsEmbedderGemma3 embedder(models, tokenizer, "CPU", {});
        const auto prefill = embedder.get_position_ids(3, 0).first;
        const auto decode = embedder.get_generation_phase_position_ids(1, 3, 0).first;
        EXPECT_EQ(prefill.data<int64_t>()[0], offset);
        EXPECT_EQ(prefill.data<int64_t>()[2], 2 + offset);
        EXPECT_EQ(decode.data<int64_t>()[0], 3 + offset);
    }
}

TEST(GGUFMultimodal, MiniCPMPositionSectionsPreserveCanvasPositionsAndRopeDelta) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    const std::vector<int64_t> time{0, 1, 2, 2, 2, 2, 2, 2, 2, 6, 7};
    const std::vector<int64_t> height{0, 1, 1, 2, 2, 2, 3, 3, 3, 6, 7};
    const std::vector<int64_t> width{0, 1, 1, 2, 3, 4, 2, 3, 4, 6, 7};
    const std::string prompt = "a<start_of_image><|image|><|image|><|image|><|image|><|image|><|image|><end_of_image>b";
    EncodedImage image{ov::Tensor(ov::element::f32, {6, 4})};
    std::fill_n(image.resized_source.data<float>(), image.resized_source.get_size(), 0.5f);
    image.crop_sizes = {{2, 3}};
    for (size_t sections : {3u, 4u}) {
        SCOPED_TRACE(sections);
        auto models = gemma4_models();
        nlohmann::json config{{"model_type", "minicpmv4_7"}, {"hidden_size", 4}};
        if (sections == 4)
            config["position_ids_sections"] = 4;
        models.config = VLMConfig(config);
        EXPECT_EQ(models.config.position_ids_sections, sections);
        models.config.image_pad_token = "<|image|>";
        models.config.video_pad_token = "<|video|>";
        models.config.vision_config_window_kernel_size = 1;
        models.config.merge_kernel_size = 1;
        InputsEmbedderMiniCPMv4_7 embedder(models, tokenizer, "CPU", {});
        embedder.set_apply_chat_template_status(false);
        VLMPerfMetrics metrics;
        const auto embeddings = embedder.get_inputs_embeds(prompt, {image}, metrics, true, {0});
        ASSERT_EQ(embeddings.get_shape(), (ov::Shape{1, 11, 4}));
        const auto [prefill, rope_delta] = embedder.get_position_ids(11, 0);
        ASSERT_EQ(prefill.get_shape(), (ov::Shape{sections, 1, 11}));
        ASSERT_EQ(rope_delta, -3);
        std::vector<int64_t> expected;
        if (sections == 4)
            expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        for (const auto& channel : {time, height, width})
            expected.insert(expected.end(), channel.begin(), channel.end());
        EXPECT_EQ(std::vector<int64_t>(prefill.data<int64_t>(), prefill.data<int64_t>() + prefill.get_size()),
                  expected);
        embedder.set_position_ids(prefill);
        for (size_t repeat = 0; repeat < 2; ++repeat) {
            const auto roundtrip = embedder.get_position_ids(11, 0);
            EXPECT_EQ(roundtrip.first.get_shape(), prefill.get_shape());
            EXPECT_EQ(roundtrip.second, rope_delta);
            EXPECT_EQ(std::vector<int64_t>(roundtrip.first.data<int64_t>(),
                                           roundtrip.first.data<int64_t>() + roundtrip.first.get_size()),
                      expected);
        }
        for (size_t tokens : {1u, 3u}) {
            const auto [decode, decode_delta] = embedder.get_generation_phase_position_ids(tokens, 11, *rope_delta);
            ASSERT_EQ(decode.get_shape(), (ov::Shape{sections, 1, tokens}));
            EXPECT_EQ(decode_delta, rope_delta);
            for (size_t channel = 0; channel < sections; ++channel)
                for (size_t token = 0; token < tokens; ++token)
                    EXPECT_EQ(decode.data<int64_t>()[channel * tokens + token],
                              (sections == 4 && channel == 0 ? 11 : 8) + token);
        }
        const auto cached = embedder.get_position_ids(3, 11);
        const auto decode = embedder.get_generation_phase_position_ids(3, 11, *rope_delta);
        EXPECT_EQ(cached.second, rope_delta);
        EXPECT_EQ(
            std::vector<int64_t>(cached.first.data<int64_t>(), cached.first.data<int64_t>() + cached.first.get_size()),
            std::vector<int64_t>(decode.first.data<int64_t>(), decode.first.data<int64_t>() + decode.first.get_size()));
    }
}

TEST(GGUFMultimodal, MiniCPMLearnedPositionBucketsPreserveProcessorResize) {
    using namespace ov::genai;
    auto models = gemma4_models();
    models.config = VLMConfig(nlohmann::json{{"model_type", "minicpmv4_7"},
                                             {"hidden_size", 1},
                                             {"vision_config", {{"image_size", 980}, {"window_kernel_size", {2}}}},
                                             {"merge_kernel_size", {2}}});
    models.processor_config.patch_size = 14;
    models.processor_config.image_size = 448;
    models.processor_config.scale_resolution = 448;
    models.processor_config.max_slice_nums = 1;
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 3, 14, -1});
    pixels->output(0).set_names({"pixel_values"});
    auto positions = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    positions->output(0).set_names({"position_ids"});
    auto window = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    window->output(0).set_names({"window_index"});
    auto merge = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
    merge->output(0).set_names({"merge_index"});
    std::vector<int64_t> indices;
    for (int64_t y = 0; y < 32; y += 4)
        for (int64_t x = 0; x < 32; x += 4)
            indices.push_back(y * 32 + x);
    auto selected = std::make_shared<ov::op::v8::Gather>(positions,
                                                         ov::op::v0::Constant::create(ov::element::i64, {64}, indices),
                                                         ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto output =
        std::make_shared<ov::op::v1::Reshape>(std::make_shared<ov::op::v0::Convert>(selected, ov::element::f32),
                                              ov::op::v0::Constant::create(ov::element::i64, {2}, {64, 1}),
                                              false);
    models.models["vision_embeddings"] =
        std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{pixels, positions, window, merge});
    VisionEncoderMiniCPMv4_7 encoder(models, "CPU", {});
    ov::Tensor image(ov::element::u8, {1, 448, 448, 3});
    std::fill_n(image.data<uint8_t>(), image.get_size(), 128);
    const auto encoded = encoder.encode(image, {});
    ASSERT_EQ(encoded.crop_sizes.size(), 1);
    EXPECT_EQ(encoded.crop_sizes[0].height, 32);
    EXPECT_EQ(encoded.crop_sizes[0].width, 32);
    ASSERT_EQ(encoded.resized_source.get_shape(), (ov::Shape{64, 1}));
    const std::vector<int64_t> buckets{0, 8, 17, 26, 35, 43, 52, 61};
    for (size_t y = 0; y < buckets.size(); ++y)
        for (size_t x = 0; x < buckets.size(); ++x)
            EXPECT_EQ(encoded.resized_source.data<const float>()[y * buckets.size() + x],
                      float(buckets[y] * 70 + buckets[x]));
}

TEST(GGUFMultimodal, Gemma3nPaddingPlaceholderPreservesImageAndPerLayerSpans) {
    using namespace ov::genai;
    auto tokenizer_config = sentencepiece_config();
    auto tokens = tokenizer_config.config.at("tokens").as<std::vector<std::string>>();
    std::swap(tokens[0], tokens[3]);
    tokenizer_config.config["tokens"] = tokens;
    auto types = tokenizer_config.config.at("token_type").as<ov::Tensor>();
    std::swap(types.data<int32_t>()[0], types.data<int32_t>()[3]);
    tokenizer_config.config.at("padding_token_id").as<ov::Tensor>().data<uint32_t>()[0] = 0;
    tokenizer_config.config.at("unknown_token_id").as<ov::Tensor>().data<uint32_t>()[0] = 3;
    GGUFTokenizer tokenizer(tokenizer_config);
    EXPECT_EQ(std::count(tokens.begin(), tokens.end(), "<image_soft_token>"), 0);
    const auto placeholder = tokenizer.encode("<pad>", add_special_tokens(false)).input_ids;
    ASSERT_EQ(placeholder.get_size(), 1);
    EXPECT_EQ(placeholder.data<int64_t>()[0], 0);
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::GEMMA3N;
    models.config.image_soft_token = "<pad>";
    const size_t layers = 2, width = 3;
    std::vector<float> rows(vocab_size * layers * width);
    for (size_t token = 0; token < vocab_size; ++token)
        std::fill_n(rows.data() + token * layers * width, layers * width, float(token));
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto lookup = std::make_shared<ov::op::v8::Gather>(
        ov::op::v0::Constant::create(ov::element::f32, {vocab_size, layers * width}, rows),
        ids,
        ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto per_layer = std::make_shared<ov::op::v1::Reshape>(
        lookup,
        ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{0, 0, layers, width}),
        true);
    models.models["text_embeddings_per_layer"] =
        std::make_shared<ov::Model>(ov::OutputVector{per_layer}, ov::ParameterVector{ids});
    InputsEmbedderGemma3n embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    std::vector<EncodedImage> images;
    for (const auto value : {7.f, 11.f}) {
        images.push_back(EncodedImage{ov::Tensor(ov::element::f32, {1, 256, 4})});
        std::fill_n(images.back().resized_source.data<float>(), images.back().resized_source.get_size(), value);
    }
    const auto normalized = embedder.normalize_prompt("a<ov_genai_image_1><ov_genai_image_0>b", 0, images);
    EXPECT_EQ(normalized.images_sequence, (std::vector<size_t>{1, 0}));
    const auto encoded = tokenizer.encode(normalized.unified_prompt).input_ids;
    EXPECT_EQ(std::count(encoded.data<int64_t>(), encoded.data<int64_t>() + encoded.get_size(), 0), 512);
    VLMPerfMetrics metrics;
    const auto embeddings =
        embedder.get_inputs_embeds(normalized.unified_prompt, images, metrics, true, normalized.images_sequence);
    ASSERT_EQ(embeddings.get_shape(), (ov::Shape{1, encoded.get_size(), 4}));
    const auto& extra = embedder.get_lm_extra_inputs().at("per_layer_inputs");
    ASSERT_EQ(extra.get_shape(), (ov::Shape{1, encoded.get_size(), layers, width}));
    size_t image_tokens = 0;
    for (size_t token = 0; token < encoded.get_size(); ++token) {
        const auto token_id = encoded.data<const int64_t>()[token];
        if (token_id == 0) {
            const auto expected = image_tokens++ < 256 ? 11.f : 7.f;
            for (size_t channel = 0; channel < 4; ++channel)
                EXPECT_EQ(embeddings.data<const float>()[token * 4 + channel], expected);
        } else {
            for (size_t channel = 0; channel < 4; ++channel)
                EXPECT_EQ(embeddings.data<const float>()[token * 4 + channel], 0.25f);
        }
        for (size_t channel = 0; channel < layers * width; ++channel)
            EXPECT_EQ(extra.data<const float>()[token * layers * width + channel], float(token_id));
    }
    EXPECT_EQ(image_tokens, 512);
}

TEST(GGUFMultimodal, BoundedResizeKeepsNativeAreaBetweenTokenLimits) {
    EXPECT_EQ(bounded_image_size(488, 640, 48, 70 * 48 * 48, 1120 * 48 * 48), (std::pair<size_t, size_t>{480, 624}));
    const auto small = bounded_image_size(1, 100, 48, 70 * 48 * 48, 1120 * 48 * 48);
    EXPECT_GT(small.first, 0);
    EXPECT_GT(small.second, 0);
    EXPECT_EQ(small.first % 48, 0);
    EXPECT_EQ(small.second % 48, 0);
    EXPECT_ANY_THROW(bounded_image_size(0, 100, 48, 1, 100));
}

TEST(GGUFMultimodal, Gemma3ImageSeparatorFollowsModelConfig) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::GEMMA3;
    models.config.image_soft_token = "<pad>";
    EncodedImage image{ov::Tensor(ov::element::f32, {1, 2, 4})};
    for (const std::string separator : {std::string{}, std::string{"\n\n"}}) {
        models.config.image_separator = separator;
        InputsEmbedderGemma3 embedder(models, tokenizer, "CPU", {});
        const auto prompt = embedder.normalize_prompt("<ov_genai_image_0>a", 0, {image});
        EXPECT_EQ(prompt.unified_prompt, separator + "<start_of_image><pad><pad><end_of_image>" + separator + "a");
    }
}

TEST(GGUFMultimodal, MuseVideoKeepsFramesWhenSamplingIsDisabled) {
    using namespace ov::genai;
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::MUSE_GLIMMER;
    models.processor_config.patch_size = 2;
    models.processor_config.merge_size = 2;
    models.processor_config.temporal_patch_size = 1;
    models.video_processor_config.patch_size = 2;
    models.video_processor_config.merge_size = 2;
    models.video_processor_config.temporal_patch_size = 1;
    models.video_processor_config.do_sample_frames = false;
    auto pixels = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 12});
    pixels->output(0).set_names({"pixel_values"});
    auto grid = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, 3});
    grid->output(0).set_names({"image_grid_thw"});
    models.models["vision_embeddings"] =
        std::make_shared<ov::Model>(ov::OutputVector{pixels}, ov::ParameterVector{pixels, grid});
    InputsEmbedderMuseGlimmer embedder(models, GGUFTokenizer(sentencepiece_config()), "CPU", {});
    ov::Tensor video(ov::element::u8, {3, 4, 4, 3});
    std::memset(video.data(), 0, video.get_byte_size());
    VideoMetadata metadata;
    metadata.fps = 2.f;
    const auto encoded = embedder.encode_videos({video}, {metadata});
    ASSERT_EQ(encoded.size(), 1);
    EXPECT_EQ(encoded[0].frame_num, 3);
    EXPECT_EQ(encoded[0].metadata.frames_indices, (std::vector<size_t>{0, 1, 2}));
    EXPECT_FLOAT_EQ(encoded[0].metadata.fps, 2.f);
    const auto assumed_fps = embedder.encode_videos({video});
    EXPECT_EQ(assumed_fps[0].frame_num, 3);
    EXPECT_GT(assumed_fps[0].metadata.fps, 0.f);
}

TEST(GGUFMultimodal, ImageOnlyProcessorConfigIsLoadedForBothModalities) {
    using namespace ov::genai;
    struct ConfigEncoder : VisionEncoder {
        explicit ConfigEncoder(const std::filesystem::path& path) : VisionEncoder(path, ConfigOnlyTag{}) {}
        EncodedImage encode(const ov::Tensor&, const ov::AnyMap&) override {
            return {};
        }
    };
    ConfigEncoder encoder(std::filesystem::path(__FILE__).parent_path() / "data" / "image_only_processor");
    const auto image = encoder.get_processor_config();
    const auto video = encoder.get_video_processor_config();
    EXPECT_EQ(image.size_height, 32);
    EXPECT_EQ(image.size_width, 48);
    EXPECT_EQ(video.size_height, image.size_height);
    EXPECT_EQ(video.size_width, image.size_width);
    EXPECT_EQ(video.patch_size, image.patch_size);
}

TEST(GGUFMultimodal, LFM2AudioPlacementPreservesAdjacentAndReversedClips) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::LFM2_AUDIO;
    models.models.erase("vision_embeddings");
    InputsEmbedderLFM2Audio embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder(first_sample_audio_features);
    const auto sample = [](float value) {
        ov::Tensor audio(ov::element::f32, {1});
        audio.data<float>()[0] = value;
        return audio;
    };
    const auto encoded = embedder.encode_audios({sample(7.f), sample(9.f)});
    const auto placed = [&](const NormalizedPrompt& prompt, size_t base) {
        auto sequence = prompt.audios_sequence;
        vlm_utils::rebase_media_sequence(sequence, base);
        VLMPerfMetrics metrics;
        const auto features =
            embedder.get_inputs_embeds(prompt.unified_prompt, {}, {}, encoded, metrics, true, {}, {}, sequence, {});
        std::vector<float> values;
        for (size_t i = 0; i < features.get_size(); i += 4)
            if (features.data<const float>()[i] != 0.25f)
                values.push_back(features.data<const float>()[i]);
        return values;
    };
    const auto native = embedder.normalize_prompt("a<audio><audio>b", 0, 0, 0, {}, {}, encoded);
    EXPECT_EQ(native.audios_sequence, (std::vector<size_t>{0, 1}));
    EXPECT_EQ(native.unified_prompt, "a<|audio_start|><|audio_start|><|audio_start|><|audio_start|>b");
    EXPECT_EQ(placed(native, 0), (std::vector<float>{7.f, 7.f, 9.f, 9.f}));
    const auto reversed = embedder.normalize_prompt("<ov_genai_audio_3><ov_genai_audio_2>", 0, 0, 2, {}, {}, encoded);
    EXPECT_EQ(reversed.audios_sequence, (std::vector<size_t>{3, 2}));
    EXPECT_EQ(placed(reversed, 2), (std::vector<float>{9.f, 9.f, 7.f, 7.f}));
    EXPECT_EQ(embedder.normalize_prompt("ab", 0, 0, 0, {}, {}, {}).unified_prompt, "ab");
    EXPECT_ANY_THROW(embedder.normalize_prompt("<ov_genai_audio_0>", 0, 0, 0, {}, {}, {}));
    EXPECT_ANY_THROW(embedder.normalize_prompt("<|audio_start|><audio><audio>", 0, 0, 0, {}, {}, encoded));
    EXPECT_ANY_THROW(embedder.encode_audios({ov::Tensor(ov::element::f32, {0})}));
    EXPECT_ANY_THROW(embedder.encode_images({ov::Tensor(ov::element::u8, {1, 2, 2, 3})}));
    EXPECT_TRUE(embedder.encode_audios({}).empty());
}

TEST(GGUFMultimodal, LFM2AudioPreprocessingMatchesLiquidReference) {
    using namespace ov::opset15;
    using namespace ov::genai;
    const auto directory = std::filesystem::path(__FILE__).parent_path() / "data" / "lfm2_audio";
    auto features = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 1, 128, -1});
    features->output(0).set_names({"features"});
    auto positions = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 512});
    positions->output(0).set_names({"pos_emb"});
    auto transposed = std::make_shared<Transpose>(features, Constant::create(ov::element::i64, {4}, {0, 1, 3, 2}));
    auto output = std::make_shared<Reshape>(transposed, Constant::create(ov::element::i64, {3}, {1, -1, 128}), false);
    auto dependency = std::make_shared<Multiply>(
        std::make_shared<ReduceSum>(positions, Constant::create(ov::element::i64, {4}, {0, 1, 2, 3}), false),
        Constant::create(ov::element::f32, {}, {0.f}));
    auto result = std::make_shared<Add>(output, dependency);
    result->output(0).set_names({"audio_features"});
    auto models = gemma4_models();
    models.models["audio_embeddings"] =
        std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{features, positions});
    models.audio_encoder = AudioEncoderConfig{AudioEncoderType::LFM2, 128};
    models.config.model_type = VLMModelType::LFM2_AUDIO;
    models.config.hidden_size = 128;
    models.models.erase("vision_embeddings");
    auto ids = std::make_shared<Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto text = std::make_shared<Gather>(Constant::create(ov::element::f32, {vocab_size, 128}, {0.25f}),
                                         ids,
                                         Constant::create(ov::element::i64, {}, {0}));
    models.models["text_embeddings"] = std::make_shared<ov::Model>(ov::OutputVector{text}, ov::ParameterVector{ids});
    InputsEmbedder embedder(models,
                            GGUFTokenizer(sentencepiece_config()),
                            "CPU",
                            {{ov::hint::inference_precision.name(), ov::element::f32}});
    for (size_t count : {160, 1281, 16000}) {
        const auto load = [&](const std::string& name, void* data, size_t bytes) {
            std::ifstream stream(directory / name, std::ios::binary);
            ASSERT_TRUE(stream.is_open()) << name;
            ASSERT_TRUE(bool(stream.read(static_cast<char*>(data), bytes))) << name;
            EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
        };
        ov::Tensor waveform(ov::element::f32, {count});
        load("waveform-" + std::to_string(count) + ".bin", waveform.data(), waveform.get_byte_size());
        const auto frames = count / 160 + 1;
        std::vector<float> reference(128 * frames);
        load("features-" + std::to_string(count) + ".bin", reference.data(), reference.size() * sizeof(float));
        const auto encoded = embedder.encode_audios({waveform});
        ASSERT_EQ(encoded.size(), 1);
        ASSERT_EQ(encoded[0].audio_features.get_shape(), (ov::Shape{frames, 128}));
        double error = 0., energy = 0.;
        for (size_t t = 0; t < frames; ++t)
            for (size_t m = 0; m < 128; ++m) {
                const auto expected = reference[m * frames + t];
                const auto actual = encoded[0].audio_features.data<const float>()[t * 128 + m];
                error += (actual - expected) * (actual - expected);
                energy += expected * expected;
                EXPECT_NEAR(actual, expected, 3e-4f) << "samples=" << count << " frame=" << t << " mel=" << m;
            }
        EXPECT_LT(error / std::max(energy, 1.), 1e-8) << "samples=" << count;
    }
    EXPECT_ANY_THROW(embedder.encode_audios({ov::Tensor(ov::element::f32, {159})}));
    EXPECT_ANY_THROW(embedder.encode_audios({ov::Tensor(ov::element::f32, {2, 160})}));
}

TEST(GGUFMultimodal, LFM2VisionTiledMarkersAndFeaturesFollowPromptOrder) {
    using namespace ov::genai;
    GGUFTokenizer tokenizer(sentencepiece_config());
    auto models = gemma4_models();
    models.config.model_type = VLMModelType::LFM2_VL;
    InputsEmbedderLFM2Vision embedder(models, tokenizer, "CPU", {});
    embedder.set_apply_chat_template_status(false);
    const auto image = [](float value, size_t tokens) {
        EncodedImage result;
        result.resized_source = ov::Tensor(ov::element::f32, {1, tokens, 4});
        std::fill_n(result.resized_source.data<float>(), result.resized_source.get_size(), value);
        result.num_image_tokens = tokens;
        return result;
    };
    auto tiled = image(7.f, 6);
    tiled.slices_grid = {1, 2};
    tiled.crop_sizes = {{2, 1}, {2, 1}, {2, 1}};
    auto overview = image(9.f, 1);
    overview.crop_sizes = {{1, 1}};
    const std::vector<EncodedImage> images{tiled, overview};
    const auto normalized = embedder.normalize_prompt("<ov_genai_image_3>a<ov_genai_image_2>", 2, images);
    EXPECT_EQ(normalized.images_sequence, (std::vector<size_t>{3, 2}));
    EXPECT_EQ(normalized.unified_prompt,
              "<|image_start|><|image|><|image_end|>a<|image_start|>"
              "<|img_row_1_col_1|><|image|><|image|><|img_row_1_col_2|><|image|><|image|>"
              "<|img_thumbnail|><|image|><|image|><|image_end|>");
    VLMPerfMetrics metrics;
    auto sequence = normalized.images_sequence;
    vlm_utils::rebase_media_sequence(sequence, 2);
    const auto embeds = embedder.get_inputs_embeds(normalized.unified_prompt, images, metrics, true, sequence);
    std::vector<float> values;
    for (size_t i = 0; i < embeds.get_size(); i += 4)
        if (embeds.data<const float>()[i] != .25f)
            values.push_back(embeds.data<const float>()[i]);
    EXPECT_EQ(values, (std::vector<float>{9.f, 7.f, 7.f, 7.f, 7.f, 7.f, 7.f}));
}

TEST(GGUFTokenizerSpecialTokens, LFM2WaveformReconstructionMatchesLiquidReference) {
    using namespace ov::opset15;
    const auto directory = std::filesystem::path(__FILE__).parent_path() / "data" / "lfm2_audio";
    const auto load = [&](const std::string& name, void* data, size_t bytes) {
        std::ifstream stream(directory / name, std::ios::binary);
        ASSERT_TRUE(stream.is_open()) << name;
        ASSERT_TRUE(bool(stream.read(static_cast<char*>(data), bytes))) << name;
        EXPECT_EQ(stream.peek(), std::char_traits<char>::eof());
    };
    std::vector<float> window(1280);
    load("istft-window.bin", window.data(), window.size() * sizeof(float));
    auto input = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{-1, 641, 2});
    auto output = std::make_shared<IRDFT>(input,
                                          Constant::create(ov::element::i64, {1}, {1}),
                                          Constant::create(ov::element::i64, {1}, {1280}));
    auto request = ov::Core{}
                       .compile_model(std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{input}),
                                      "CPU",
                                      ov::hint::inference_precision(ov::element::f32))
                       .create_infer_request();
    for (size_t frames : {1, 18, 37, 18}) {
        SCOPED_TRACE(frames);
        ov::Tensor spectral(ov::element::f32, {1, frames, 1282});
        load("spectral-" + std::to_string(frames) + ".bin", spectral.data(), spectral.get_byte_size());
        std::vector<float> reference(frames * 320);
        load("istft-" + std::to_string(frames) + ".bin", reference.data(), reference.size() * sizeof(float));
        auto actual = ov::genai::lfm2_spectrum_to_waveform(spectral, window, request);
        ASSERT_EQ(actual.get_shape(), (ov::Shape{1, frames * 320}));
        double error = 0., energy = 0.;
        for (size_t i = 0; i < reference.size(); ++i) {
            const auto delta = double(actual.data<const float>()[i]) - reference[i];
            error += delta * delta;
            energy += double(reference[i]) * reference[i];
        }
        EXPECT_LT(error / energy, 1e-8);
    }
}

TEST(GGUFMultimodal, LFM2VisionNormalizesPixelsAndPadsTiledCanvas) {
    using namespace ov::genai;
    using namespace ov::opset15;
    const auto models_for_pixels = [] {
        auto models = gemma4_models();
        models.config.model_type = VLMModelType::LFM2_VL;
        models.processor_config.patch_size = 1;
        models.processor_config.merge_size = 1;
        models.processor_config.min_pixels = 1;
        models.processor_config.max_pixels = 16;
        models.processor_config.tile_size = 4;
        models.processor_config.min_patches = 2;
        models.processor_config.max_patches = 2;
        models.processor_config.image_mean = {0.5f, 0.5f, 0.5f};
        models.processor_config.image_std = {0.5f, 0.5f, 0.5f};
        auto pixels = std::make_shared<Parameter>(ov::element::f32, ov::PartialShape{1, 3, -1, -1});
        pixels->output(0).set_names({"pixel_values"});
        auto indices = std::make_shared<Parameter>(ov::element::i32, ov::PartialShape{1, 1, 1, -1});
        indices->output(0).set_names({"merge_indices"});
        auto flat = std::make_shared<Reshape>(pixels, Constant::create(ov::element::i64, {3}, {1, 3, -1}), false);
        auto output = std::make_shared<Transpose>(flat, Constant::create(ov::element::i64, {3}, {0, 2, 1}));
        auto dependency = std::make_shared<Multiply>(
            std::make_shared<Convert>(
                std::make_shared<ReduceSum>(indices, Constant::create(ov::element::i64, {4}, {0, 1, 2, 3}), false),
                ov::element::f32),
            Constant::create(ov::element::f32, {}, {0.f}));
        auto result = std::make_shared<Add>(output, dependency);
        models.models["vision_embeddings"] =
            std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{pixels, indices});
        return models;
    };
    VisionEncoderLFM2 encoder(models_for_pixels(), "CPU", {{ov::hint::inference_precision.name(), ov::element::f32}});
    ov::Tensor image(ov::element::u8, {1, 2, 3, 3});
    for (size_t i = 0; i < 6; ++i) {
        image.data<uint8_t>()[i * 3] = 255;
        image.data<uint8_t>()[i * 3 + 1] = 0;
        image.data<uint8_t>()[i * 3 + 2] = 128;
    }
    const auto encoded = encoder.encode(image);
    ASSERT_EQ(encoded.num_image_tokens, 6);
    ASSERT_EQ(encoded.resized_source.get_shape(), (ov::Shape{1, 6, 3}));
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(encoded.resized_source.data<const float>()[i * 3], 1.f);
        EXPECT_FLOAT_EQ(encoded.resized_source.data<const float>()[i * 3 + 1], -1.f);
        EXPECT_NEAR(encoded.resized_source.data<const float>()[i * 3 + 2], 1.f / 255.f, 1e-7f);
    }
    ov::Tensor tiled(ov::element::u8, {1, 4, 12, 3});
    std::fill_n(tiled.data<uint8_t>(), tiled.get_size(), 255);
    auto output = encoder.encode(tiled);
    ASSERT_TRUE(output.slices_grid.has_slices());
    ASSERT_EQ(output.crop_sizes.size(), 3);
    ASSERT_EQ(output.crop_sizes[0].height, 16);
    ASSERT_EQ(output.crop_sizes[1].height, 16);
    // The 12x4 source is resized to 8x3 and padded to the 8x4 canvas.
    for (size_t tile = 0; tile < 2; ++tile)
        for (size_t y = 0; y < 4; ++y)
            for (size_t x = 0; x < 4; ++x)
                for (size_t c = 0; c < 3; ++c)
                    EXPECT_FLOAT_EQ(output.resized_source.data<const float>()[((tile * 16 + y * 4 + x) * 3) + c],
                                    y < 3 ? 1.f : -1.f);
}

TEST(GGUFTokenizer, MetadataRejectsModelsWithoutTokenizerInfo) {
    EXPECT_THROW(ov::genai::GGUFTokenizerParameters::from_model(nullptr), ov::Exception);
}
