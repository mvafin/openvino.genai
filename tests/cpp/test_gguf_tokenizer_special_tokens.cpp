// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "gtest/gtest.h"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/vlm_chat_context.hpp"

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
    for (const auto* token : {"<|audio|>", "<|audio>", "<audio|>", "<|image|>", "<|video|>"}) {
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
}  // namespace

TEST(GGUFTokenizer, SentencePieceControlTokensUseTheirVocabularyIds) {
    ov::genai::Tokenizer tokenizer(sentencepiece_config());
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
    ov::genai::Tokenizer tokenizer(config);
    EXPECT_EQ(tokenizer.apply_chat_template({{{"role", "user"}, {"content", "a"}}}, false), "firstseconda");
}

TEST(GGUFMultimodal, GemmaAudioHistoryAndResetPreserveFeaturePlacement) {
    using namespace ov::genai;
    Tokenizer tokenizer(sentencepiece_config());
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto table = ov::op::v0::Constant::create(ov::element::f32, {272, 4}, {0.25f});
    auto lookup =
        std::make_shared<ov::op::v8::Gather>(table, ids, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto embedding_model = std::make_shared<ov::Model>(ov::OutputVector{lookup}, ov::ParameterVector{ids});
    auto embeddings = std::make_shared<EmbeddingsModel>(embedding_model, "CPU", ov::AnyMap{});
    VLMConfig config;
    config.hidden_size = 4;
    config.video_token = "<|video|>";
    InputsEmbedderGemma4 embedder(config, tokenizer, nullptr, embeddings, "CPU", false);
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder([](const ov::Tensor& audio) {
        ov::Tensor features(ov::element::f32, {1, 2, 4});
        std::fill_n(features.data<float>(), features.get_size(), audio.data<const float>()[0]);
        return std::vector<ov::Tensor>{features};
    });
    const auto encode = [&](float value, bool chat) {
        ov::Tensor audio(ov::element::f32, {1});
        audio.data<float>()[0] = value;
        embedder.encode_audios({audio}, chat);
        return embedder.normalize_prompt("a<|audio|>b", 0, {}).unified_prompt;
    };
    const auto count_features = [&](const std::string& prompt, float value) {
        VLMPerfMetrics metrics;
        const auto features = embedder.get_inputs_embeds(prompt, {}, metrics);
        return std::count(features.data<const float>(), features.data<const float>() + features.get_size(), value);
    };
    const auto first = encode(7.f, true);
    const auto second = encode(9.f, true);
    EXPECT_EQ(count_features(first + second, 7.f), 8);
    EXPECT_EQ(count_features(first + second, 9.f), 8);
    embedder.encode_audios({}, true);
    EXPECT_EQ(embedder.normalize_prompt("ab", 0, {}).unified_prompt, "ab");
    EXPECT_EQ(count_features(first + second + "ab", 7.f), 8);
    encode(13.f, true);
    embedder.update_chat_history("", GenerationStatus::CANCEL);
    EXPECT_EQ(count_features(first + second, 9.f), 8);
    EXPECT_EQ(count_features(first + second, 13.f), 0);
    embedder.finish_chat();
    const auto independent = encode(11.f, false);
    EXPECT_EQ(count_features(independent, 11.f), 8);
    embedder.encode_audios({}, false);
    EXPECT_EQ(count_features("ab", 11.f), 0);
}

TEST(GGUFMultimodal, ModernAudioHistorySwitchEditAndRollback) {
    using namespace ov::genai;
    Tokenizer tokenizer(sentencepiece_config());
    auto ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    ids->output(0).set_names({"input_ids"});
    auto table = ov::op::v0::Constant::create(ov::element::f32, {272, 4}, {0.25f});
    auto lookup =
        std::make_shared<ov::op::v8::Gather>(table, ids, ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto embedding_model = std::make_shared<ov::Model>(ov::OutputVector{lookup}, ov::ParameterVector{ids});
    auto embeddings = std::make_shared<EmbeddingsModel>(embedding_model, "CPU", ov::AnyMap{});
    VLMConfig config;
    config.model_type = VLMModelType::GEMMA4;
    config.hidden_size = 4;
    config.video_token = "<|video|>";
    InputsEmbedder embedder(config, tokenizer, nullptr, embeddings, "CPU", false);
    embedder.set_apply_chat_template_status(false);
    embedder.set_audio_encoder([](const ov::Tensor& audio) {
        ov::Tensor features(ov::element::f32, {1, 2, 4});
        std::fill_n(features.data<float>(), features.get_size(), audio.data<const float>()[0]);
        return std::vector<ov::Tensor>{features};
    });
    auto registry = std::make_shared<VisionRegistry>();
    const auto audio = [](float value) {
        ov::Tensor tensor(ov::element::f32, {1});
        tensor.data<float>()[0] = value;
        return tensor;
    };
    const auto count_features = [&](const ChatHistory& history, float value) {
        std::string prompt;
        for (size_t i = 0; i < history.size(); ++i)
            prompt += history[i]["content"].get_string();
        VLMPerfMetrics metrics;
        const auto features = embedder.get_inputs_embeds(prompt, {}, {}, metrics);
        return std::count(features.data<const float>(), features.data<const float>() + features.get_size(), value);
    };
    ChatHistory first({{{"role", "user"}, {"content", "a<|audio|>b"}}});
    auto first_data = VLMChatContext(first, registry, embedder).process({}, {}, {}, {audio(7.f)});
    EXPECT_EQ(count_features(first_data.normalized_history, 7.f), 8);
    ChatHistory second({{{"role", "user"}, {"content", "<|audio|>"}}});
    auto second_data = VLMChatContext(second, registry, embedder).process({}, {}, {}, {audio(9.f)});
    EXPECT_EQ(count_features(second_data.normalized_history, 9.f), 8);
    first.push_back({{"role", "assistant"}, {"content", "a"}});
    first.push_back({{"role", "user"}, {"content", "b"}});
    first_data = VLMChatContext(first, registry, embedder).process({});
    EXPECT_EQ(count_features(first_data.normalized_history, 7.f), 8);
    EXPECT_EQ(count_features(first_data.normalized_history, 9.f), 0);
    first.push_back({{"role", "user"}, {"content", "<|audio|>"}});
    VLMChatContext cancelled(first, registry, embedder);
    auto cancelled_data = cancelled.process({}, {}, {}, {audio(11.f)});
    EXPECT_EQ(count_features(cancelled_data.normalized_history, 11.f), 8);
    cancelled.rollback();
    first.pop_back();
    first_data = VLMChatContext(first, registry, embedder).process({});
    EXPECT_EQ(count_features(first_data.normalized_history, 7.f), 8);
    EXPECT_EQ(count_features(first_data.normalized_history, 11.f), 0);
    first[0]["content"] = "b<|audio|>a";
    first.pop_back();
    first.pop_back();
    first_data = VLMChatContext(first, registry, embedder).process({}, {}, {}, {audio(13.f)});
    EXPECT_EQ(count_features(first_data.normalized_history, 13.f), 8);
    EXPECT_EQ(count_features(first_data.normalized_history, 7.f), 0);
    first[0]["content"] = "a<|audio|>";
    VLMChatContext edited(first, registry, embedder);
    edited.process({}, {}, {}, {audio(17.f)});
    edited.rollback();
    EXPECT_TRUE(ChatHistoryInternalState::get_or_create(first, registry)->get_messages_metadata().empty());
}
