// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/core.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"

TEST(Eagle3ModelTransforms, PreservesKvCacheParameterRtInfo) {
    const auto cache_shape = ov::PartialShape{2, 4};
    auto key_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
    key_cache->set_friendly_name("key_cache.0");
    key_cache->output(0).set_names({"key_cache.0"});
    key_cache->get_rt_info()["enable_keep_const_precision"] = true;

    auto value_cache = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, cache_shape);
    value_cache->set_friendly_name("value_cache.0");
    value_cache->output(0).set_names({"value_cache.0"});
    value_cache->get_rt_info()["enable_keep_const_precision"] = true;

    auto main_model = std::make_shared<ov::Model>(ov::OutputVector{key_cache, value_cache},
                                                  ov::ParameterVector{key_cache, value_cache});
    const auto kv_update_model = ov::genai::utils::eagle3::create_eagle3_kv_update_model(main_model);

    for (const auto& input_name : {"key_cache.0", "value_cache.0"}) {
        const auto parameter = kv_update_model->input(input_name).get_node_shared_ptr();
        ASSERT_TRUE(parameter->get_rt_info().count("enable_keep_const_precision"));
        EXPECT_TRUE(parameter->get_rt_info().at("enable_keep_const_precision").as<bool>());
    }
}

TEST(Eagle3ModelTransforms, MovesGgufEncoderAndPreservesRankThreeBoundary) {
    using namespace ov::op;
    namespace eagle = ov::genai::utils::eagle3;
    auto target_input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 4});
    target_input->set_friendly_name("blk.0.l_out");
    auto target = std::make_shared<ov::Model>(ov::OutputVector{target_input}, ov::ParameterVector{target_input});
    target->set_rt_info(std::string("qwen3"), "gguf_architecture");
    eagle::transform_hidden_state(target, {0, 0, 0});

    auto hidden = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 12});
    hidden->output(0).set_names({"hidden_states"});
    auto expanded = std::make_shared<v0::Unsqueeze>(hidden, v0::Constant::create(ov::element::i64, {1}, {1}));
    auto weights = v0::Constant::create(ov::element::f32, {4, 12}, std::vector<float>(48, 0.25f));
    auto fc = std::make_shared<v0::MatMul>(expanded, weights, false, true);
    fc->set_friendly_name("MatMul_eagle3.fc");
    auto draft = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{hidden});
    draft->set_rt_info(std::string("eagle3"), "gguf_architecture");
    eagle::move_fc_from_draft_to_main(draft, target);

    EXPECT_EQ(draft->input("hidden_states").get_partial_shape(), (ov::PartialShape{1, -1, 4}));
    EXPECT_EQ(target->output("last_hidden_state").get_partial_shape(), (ov::PartialShape{1, -1, 4}));
    auto request =
        ov::Core{}.compile_model(target, "CPU", ov::hint::inference_precision(ov::element::f32)).create_infer_request();
    auto input = ov::Tensor(ov::element::f32, {1, 1, 2, 4});
    std::fill_n(input.data<float>(), input.get_size(), 2.0f);
    request.set_input_tensor(input);
    request.infer();
    auto output = request.get_tensor("last_hidden_state");
    for (size_t i = 0; i < output.get_size(); ++i)
        EXPECT_FLOAT_EQ(output.data<const float>()[i], 6.0f);
}

TEST(Eagle3ModelTransforms, PreservesGgufDraftPrenormOutput) {
    using namespace ov::op;
    auto hidden = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 4});
    hidden->output(0).set_names({"last_hidden_state"});
    auto logits = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 32});
    logits->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{logits, hidden}, ov::ParameterVector{logits, hidden});
    model->set_rt_info(std::string("eagle3"), "gguf_architecture");
    EXPECT_NO_THROW(ov::genai::utils::eagle3::transform_hidden_state(model, {-1}));
    EXPECT_EQ(model->output("last_hidden_state").get_partial_shape(), hidden->get_output_partial_shape(0));
    auto missing = std::make_shared<ov::Model>(ov::OutputVector{logits}, ov::ParameterVector{logits});
    missing->set_rt_info(std::string("eagle3"), "gguf_architecture");
    EXPECT_THROW(ov::genai::utils::eagle3::transform_hidden_state(missing, {-1}), ov::Exception);
}

TEST(Eagle3ModelTransforms, FindsTaggedGgufEmbeddingWithSmallVocabulary) {
    using namespace ov::op;
    auto ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto weights = v0::Constant::create(ov::element::f32, {32, 4}, std::vector<float>(128, .25f));
    auto gather = std::make_shared<v8::Gather>(weights, ids, v0::Constant::create(ov::element::i64, {}, {0}));
    gather->get_rt_info()["gguf.token_embedding"] = true;
    auto model = std::make_shared<ov::Model>(ov::OutputVector{gather}, ov::ParameterVector{ids});
    EXPECT_EQ(ov::genai::utils::eagle3::find_embedding_gather(model), gather);
    gather->get_rt_info().erase("gguf.token_embedding");
    auto reshaped =
        std::make_shared<v1::Reshape>(gather, v0::Constant::create(ov::element::i64, {4}, {1, 1, -1, 4}), false);
    reshaped->get_rt_info()["gguf.token_embedding"] = true;
    model = std::make_shared<ov::Model>(ov::OutputVector{reshaped}, ov::ParameterVector{ids});
    EXPECT_EQ(ov::genai::utils::eagle3::find_embedding_gather(model), gather);
}

TEST(Eagle3ModelTransforms, AttachesBorrowedGgufWeightsAndPreservesOwnWeights) {
    using namespace ov::op;
    namespace eagle = ov::genai::utils::eagle3;
    auto ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto weights = v0::Constant::create(ov::element::f32, {32, 4}, std::vector<float>(128, .25f));
    auto gather = std::make_shared<v8::Gather>(weights, ids, v0::Constant::create(ov::element::i64, {}, {0}));
    gather->get_rt_info()["gguf.token_embedding"] = true;
    auto head = std::make_shared<v0::MatMul>(gather, weights, false, true);
    head->output(0).set_names({"logits"});
    auto target = std::make_shared<ov::Model>(ov::OutputVector{head}, ov::ParameterVector{ids});
    auto draft_ids = std::make_shared<v0::Parameter>(ov::element::i64, ov::PartialShape{1, -1});
    auto embedding = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    embedding->set_friendly_name("gguf.target_token_embedding");
    auto output = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 4});
    output->set_friendly_name("gguf.target_lm_head");
    auto draft_gather =
        std::make_shared<v8::Gather>(embedding, draft_ids, v0::Constant::create(ov::element::i64, {}, {0}));
    auto draft_head = std::make_shared<v0::MatMul>(draft_gather, output, false, true);
    auto draft =
        std::make_shared<ov::Model>(ov::OutputVector{draft_head}, ov::ParameterVector{draft_ids, embedding, output});
    draft->set_rt_info(std::string("eagle3"), "gguf_architecture");
    eagle::share_vocabulary(target, draft);
    EXPECT_EQ(draft->get_parameters().size(), 1u);
    EXPECT_EQ(draft->output().get_partial_shape(), (ov::PartialShape{1, -1, 32}));
    EXPECT_NE(draft_gather->input_value(0).get_node(), weights.get());
    auto own = draft_gather->input_value(0).get_node_shared_ptr();
    eagle::share_vocabulary(target, draft);
    EXPECT_EQ(draft_gather->input_value(0).get_node_shared_ptr(), own);
    auto request = ov::Core{}.compile_model(draft, "CPU").create_infer_request();
    ov::Tensor input(ov::element::i64, {1, 2});
    input.data<int64_t>()[0] = 3;
    input.data<int64_t>()[1] = 7;
    request.set_input_tensor(input);
    request.infer();
    const auto logits = request.get_output_tensor();
    for (size_t i = 0; i < logits.get_size(); ++i)
        EXPECT_FLOAT_EQ(logits.data<const float>()[i], .25f);
}

TEST(Eagle3ModelTransforms, UsesIdentityMappingOnlyForGgufWithoutReducedVocabulary) {
    namespace eagle = ov::genai::utils::eagle3;
    auto logits = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 8});
    logits->output(0).set_names({"logits"});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{logits}, ov::ParameterVector{logits});
    EXPECT_EQ(eagle::extract_d2t_mapping_table(model), nullptr);
    model->set_rt_info(std::string("eagle3"), "gguf_architecture");
    const auto identity = eagle::extract_d2t_mapping_table(model);
    ASSERT_NE(identity, nullptr);
    EXPECT_EQ(identity->cast_vector<int64_t>(), (std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}));
    EXPECT_EQ(model->get_results().size(), 1u);
    auto reduced = ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 3, 5, 7});
    reduced->set_friendly_name("d2t");
    model->add_results({std::make_shared<ov::op::v0::Result>(reduced)});
    EXPECT_EQ(eagle::extract_d2t_mapping_table(model), reduced);
    EXPECT_EQ(model->get_results().size(), 1u);
}
