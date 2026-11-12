// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dflash_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <openvino/pass/sdpa_to_paged_attention.hpp>
#include <string>
#include <vector>

#include "continuous_batching/paged_attention_transformations.hpp"
#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "speculative_decoding/continuous_batching/dflash_candidate_selector.hpp"
#include "utils.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

namespace {

struct DraftCandidateToken {
    int64_t token_id;
    float log_prob;
};

std::vector<float> zero_log_probs(size_t count) {
    return std::vector<float>(count, 0.0f);
}

bool has_compiled_input(const ov::CompiledModel& model, const std::string& name) {
    const auto inputs = model.inputs();
    return std::find_if(inputs.begin(), inputs.end(), [&](const ov::Output<const ov::Node>& port) {
               return port.get_names().count(name) != 0;
           }) != inputs.end();
}

bool model_has_output(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    const auto outputs = model->outputs();
    return std::find_if(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& port) {
               return port.get_names().count(name) != 0;
           }) != outputs.end();
}

void validate_target_has_no_unmanaged_state(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(model, "DFlash target model cannot be null.");
    std::vector<std::string> unmanaged_state_ops;
    for (const auto& op : model->get_ordered_ops()) {
        const std::string type_name = op->get_type_name();
        if (type_name == "ReadValue" || type_name == "Assign") {
            unmanaged_state_ops.push_back(op->get_friendly_name());
        }
    }
    OPENVINO_ASSERT(unmanaged_state_ops.empty(),
                    "DFlash CB/PA target model contains unmanaged state op(s) after paging conversion. ",
                    "First unmanaged op: ",
                    unmanaged_state_ops.empty() ? std::string{} : unmanaged_state_ops.front(),
                    ". Convert KV, conv, and GatedDeltaNet state to managed paging before enabling DFlash.");
}

// The target decodes a request alone with num_assistant_tokens == 0, and an echo-only request
// (max_new_tokens == 0) generates nothing to draft.
bool has_draft_request(const GenerationConfig& config) {
    return config.num_assistant_tokens.value_or(0) > 0 && config.max_new_tokens > 0;
}

}  // namespace

class ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashCBDraftRunner {
public:
    DFlashCBDraftRunner(const ov::genai::ModelDesc& model_desc,
                        const Tokenizer& tokenizer,
                        const ov::genai::utils::dflash::DFlashRTInfo& rt_info,
                        EmbeddingsModel::Ptr embedding_model = nullptr)
        : m_tokenizer(tokenizer),
          m_embedding_model(std::move(embedding_model)),
          m_sampler(tokenizer),
          m_mask_token_id(rt_info.mask_token_id),
          m_candidate_position_offset(rt_info.candidate_position_offset) {
        if (model_desc.model->has_rt_info("gguf.dflash_selector_top_k")) {
            m_selector_top_k = model_desc.model->get_rt_info<int>("gguf.dflash_selector_top_k");
            m_selector_block_size = model_desc.model->get_rt_info<int>("gguf.dflash_block_size");
            m_candidate_position_offset = 1;
        }
        if (model_desc.model->has_rt_info("gguf.lilicorr_top_k")) {
            m_selector_top_k = model_desc.model->get_rt_info<int>("gguf.lilicorr_top_k");
            m_selector_block_size = model_desc.model->get_rt_info<int>("gguf.dflash_block_size");
            m_candidate_position_offset = 1;
        }
        if (model_desc.model->has_rt_info("gguf.dflash_markov")) {
            m_selector_block_size = model_desc.model->get_rt_info<int>("gguf.dflash_block_size");
            m_candidate_position_offset = model_desc.model->get_rt_info<bool>("gguf.dflash_sample_from_anchor") ? 0 : 1;
        }
        m_request = create_draft_infer_request(model_desc, static_cast<bool>(m_embedding_model));
        m_has_beam_idx = has_compiled_input(m_request.get_compiled_model(), "beam_idx");
        if (m_has_beam_idx) {
            m_beam_idx = ov::Tensor(ov::element::i32, {BATCH_SIZE});
            std::fill_n(m_beam_idx.data<int32_t>(), m_beam_idx.get_size(), 0);
        }
        m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.tokenization_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.detokenization_durations = {MicroSeconds(0.0f)};
    }

    void initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config) {
        const auto shape = input_ids.get_shape();
        OPENVINO_ASSERT(shape.size() == 2 && shape[0] == BATCH_SIZE && shape[1] > 0,
                        "Expected DFlash input_ids shape [1, seq_len].");
        const int64_t* ids_data = input_ids.data<const int64_t>();
        TokenIds prompt_ids(ids_data, ids_data + shape[1]);
        initialize_sampler_sequence(std::move(prompt_ids), config);
    }

    void initialize_sequence(size_t prompt_length, const GenerationConfig& config) {
        initialize_sampler_sequence(
            dflash_cb::build_placeholder_prompt_ids(prompt_length, m_tokenizer.get_pad_token_id()),
            config);
    }

    void sync_generated_tokens(const std::vector<int64_t>& target_generated_tokens) {
        auto seq = (*m_sequence_group)[0];
        if (seq->get_generated_len() > 0) {
            seq->remove_last_tokens(seq->get_generated_len());
        }
        for (auto token : target_generated_tokens) {
            seq->append_token(token, 0.0f);
        }
        m_sequence_group->update_processed_tokens_num(m_prompt_length + target_generated_tokens.size());
        seq->set_status(SequenceStatus::RUNNING);
    }

    ov::Tensor infer(int64_t seed_token, const ov::Tensor& hidden_delta, size_t candidate_count) {
        OPENVINO_ASSERT(hidden_delta && hidden_delta.get_size() > 0, "DFlash hidden delta must be provided.");
        const auto hidden_delta_shape = hidden_delta.get_shape();
        OPENVINO_ASSERT(hidden_delta_shape.size() == 3 && hidden_delta_shape[1] == BATCH_SIZE,
                        "DFlash draft hidden_states input must have shape [seq_len, 1, hidden].");
        const size_t hidden_delta_length = hidden_delta_shape[0];
        if (m_selector_top_k > 0)
            candidate_count = size_t(m_selector_block_size - 1);
        else if (m_selector_block_size > 0)
            candidate_count = std::min(candidate_count, size_t(m_selector_block_size) - m_candidate_position_offset);

        auto input_ids = build_input_ids(seed_token, candidate_count);
        auto position_ids = build_position_ids(hidden_delta_length, candidate_count);
        auto attention_mask = build_attention_mask(hidden_delta_length, candidate_count);
        m_request.set_tensor("hidden_states", hidden_delta);
        m_request.set_tensor("position_ids", position_ids);
        m_request.set_tensor("attention_mask", attention_mask);
        if (m_embedding_model) {
            CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(
                m_embedding_model->get_request_queue().get());
            ov::Tensor input_embeds = m_embedding_model->infer(embeddings_request_guard.get(), input_ids);
            m_request.set_tensor("inputs_embeds", input_embeds);
            // The embeddings request owns input_embeds. Keep it reserved until
            // synchronous draft inference has consumed that tensor.
            update_inference_time(execute_inference());
        } else {
            m_request.set_tensor("input_ids", input_ids);
            update_inference_time(execute_inference());
        }
        m_committed_context_length += hidden_delta_length;
        return m_request.get_tensor("logits");
    }

    std::vector<DraftCandidateToken> sample_candidates(const ov::Tensor& logits, size_t candidate_count) {
        std::vector<DraftCandidateToken> candidates;
        candidates.reserve(candidate_count);
        if (m_selector_top_k > 0) {
            const auto lattice = m_request.get_tensor("draft_lattice");
            const auto shape = lattice.get_shape();
            OPENVINO_ASSERT(
                shape.size() == 3 && shape[0] == 1 && shape[2] >= size_t(m_selector_top_k * (m_selector_top_k + 1)),
                "Invalid DFlash selector lattice shape");
            size_t predecessor = 0;
            const auto count = std::min({candidate_count, shape[1] - 1, size_t(m_selector_block_size - 1)});
            for (size_t idx = 1; idx <= count; ++idx) {
                const auto* row = lattice.data<const float>() + idx * shape[2];
                const auto* scores = row + m_selector_top_k + predecessor * m_selector_top_k;
                predecessor = std::max_element(scores, scores + m_selector_top_k) - scores;
                candidates.push_back({static_cast<int64_t>(row[predecessor]), 0.f});
            }
            m_raw_perf_metrics.m_batch_sizes.emplace_back(candidates.size());
            return candidates;
        }
        const auto shape = logits.get_shape();
        const auto offset = m_selector_block_size > 0 ? m_candidate_position_offset : 0;
        candidate_count = std::min(candidate_count, shape[1] - offset);
        for (size_t idx = offset; idx < offset + candidate_count; ++idx) {
            ov::Tensor one_position(logits, ov::Coordinate{0, idx, 0}, ov::Coordinate{1, idx + 1, shape[2]});
            auto sampled = sample_one_candidate(one_position);
            candidates.insert(candidates.end(), sampled.begin(), sampled.end());
        }
        m_raw_perf_metrics.m_batch_sizes.emplace_back(candidates.size());
        return candidates;
    }

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

    size_t get_consumed_hidden_states() const {
        return m_committed_context_length;
    }

private:
    void initialize_sampler_sequence(TokenIds prompt_ids, const GenerationConfig& config) {
        m_prompt_length = prompt_ids.size();
        // Sampler state is keyed by request_id; we reuse request_id=1, so clear per-request context.
        m_sampler.clear_request_info(1);
        m_sequence_group = std::make_shared<SequenceGroup>(1, prompt_ids, config);
        m_sequence_group->update_processed_tokens_num(m_prompt_length);
        m_committed_context_length = 0;
        m_request.reset_state();
        if (m_has_beam_idx) {
            m_request.set_tensor("beam_idx", m_beam_idx);
        }
        m_raw_perf_metrics.m_inference_durations = {MicroSeconds(0.0f)};
        m_raw_perf_metrics.m_durations.clear();
        m_raw_perf_metrics.m_batch_sizes.clear();
    }

    static ov::InferRequest create_draft_infer_request(const ov::genai::ModelDesc& model_desc,
                                                       bool use_external_embeddings) {
        OPENVINO_ASSERT(model_desc.model, "DFlash draft model cannot be null.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "hidden_states"),
                        "DFlash CB/PA draft model must have 'hidden_states' input.");
        if (use_external_embeddings) {
            OPENVINO_ASSERT(utils::has_input(model_desc.model, "inputs_embeds"),
                            "DFlash VLM draft model must have an 'inputs_embeds' input.");
            OPENVINO_ASSERT(!utils::has_input(model_desc.model, "input_ids"),
                            "DFlash VLM draft model must not have an 'input_ids' input.");
        } else {
            OPENVINO_ASSERT(utils::has_input(model_desc.model, "input_ids"),
                            "DFlash CB/PA draft model must have an 'input_ids' input after load-time transforms.");
        }
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "position_ids"),
                        "DFlash CB/PA draft model must have a 'position_ids' input.");
        OPENVINO_ASSERT(utils::has_input(model_desc.model, "attention_mask"),
                        "DFlash CB/PA draft model must have an 'attention_mask' input.");
        if (model_desc.device == "NPU") {
            auto kv_axes_pos = utils::get_kv_axes_pos(model_desc.model);
            auto npu_compile_result =
                utils::compile_decoder_for_npu(model_desc.model, model_desc.properties, kv_axes_pos);
            return npu_compile_result.first.create_infer_request();
        }
        return utils::singleton_core()
            .compile_model(model_desc.model, model_desc.device, model_desc.properties)
            .create_infer_request();
    }

    ov::Tensor build_input_ids(int64_t seed_token, size_t candidate_count) const {
        return dflash_cb::build_draft_input_ids(seed_token,
                                                m_mask_token_id,
                                                candidate_count,
                                                m_candidate_position_offset);
    }

    ov::Tensor build_position_ids(size_t hidden_delta_length, size_t candidate_count) const {
        return dflash_cb::build_draft_position_ids(m_committed_context_length,
                                                   hidden_delta_length,
                                                   candidate_count,
                                                   m_candidate_position_offset);
    }

    ov::Tensor build_attention_mask(size_t hidden_delta_length, size_t candidate_count) const {
        return dflash_cb::build_draft_attention_mask(m_committed_context_length,
                                                     hidden_delta_length,
                                                     candidate_count,
                                                     m_candidate_position_offset);
    }

    std::vector<DraftCandidateToken> sample_one_candidate(const ov::Tensor& logits) {
        const auto sequence = (*m_sequence_group)[0];
        const size_t generated_before = sequence->get_generated_len();
        m_sequence_group->schedule_tokens(1);
        m_sequence_group->set_output_seq_len(1);
        m_sequence_group->set_num_validated_tokens(0);
        m_sampler.sample({m_sequence_group}, logits, false);
        m_sequence_group->finish_iteration();

        const auto& generated = sequence->get_generated_ids();
        if (generated.size() <= generated_before) {
            return {};
        }
        const auto& log_probs = sequence->get_generated_log_probs();
        OPENVINO_ASSERT(log_probs.size() >= generated.size(), "Generated token log-probs are out of sync.");
        std::vector<DraftCandidateToken> candidates;
        candidates.reserve(generated.size() - generated_before);
        for (size_t idx = generated_before; idx < generated.size(); ++idx) {
            candidates.push_back({generated[idx], log_probs[idx]});
        }
        return candidates;
    }

    uint64_t execute_inference() {
        auto start = std::chrono::steady_clock::now();
        m_request.infer();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
    }

    void update_inference_time(uint64_t inference_time_us) {
        m_raw_perf_metrics.m_durations.emplace_back(static_cast<float>(inference_time_us));
        m_raw_perf_metrics.m_inference_durations[0] += MicroSeconds(static_cast<float>(inference_time_us));
    }

    static constexpr size_t BATCH_SIZE = 1;
    Tokenizer m_tokenizer;
    EmbeddingsModel::Ptr m_embedding_model;
    mutable ov::InferRequest m_request;
    SequenceGroup::Ptr m_sequence_group;
    Sampler m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    bool m_has_beam_idx = false;
    ov::Tensor m_beam_idx;
    size_t m_prompt_length = 0;
    size_t m_committed_context_length = 0;
    int64_t m_mask_token_id = -1;
    size_t m_candidate_position_offset = 1;
    int m_selector_top_k = 0;
    int m_selector_block_size = 0;
};

ContinuousBatchingPipeline::DFlashDecodingImpl::DFlashDecodingImpl(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const ov::genai::utils::dflash::DFlashRTInfo& rt_info)
    : m_rt_info(rt_info) {
    OPENVINO_ASSERT(m_rt_info.dflash_mode, "DFlash continuous batching requires dflash_mode=true.");
    OPENVINO_ASSERT(!m_rt_info.target_layer_ids.empty(), "DFlash target_layer_ids cannot be empty.");
    auto main_model = main_model_desc.model;
    auto draft_model = draft_model_desc.model;
    OPENVINO_ASSERT(main_model && draft_model, "DFlash requires both target and draft models.");
    const bool gguf_draft = draft_model->has_rt_info("gguf_architecture") &&
                            draft_model->get_rt_info<std::string>("gguf_architecture") == "dflash";
    OPENVINO_ASSERT(gguf_draft || utils::has_input(draft_model, "token_type_ids"),
                    "DFlash draft model has no 'token_type_ids' input. It was exported for the removed stateful "
                    "DFlash draft runner; re-export the draft with the current optimum-intel.");
    if (gguf_draft)
        utils::eagle3::attach_gguf_target_weights(main_model, draft_model);
    // TODO: support cache eviction. The draft runs with an unlimited max_num_batched_tokens, which sizes the cache
    // rotation table, and the draft windows were not validated with evicted draft KV.
    OPENVINO_ASSERT(
        !main_model_desc.scheduler_config.use_cache_eviction && !draft_model_desc.scheduler_config.use_cache_eviction,
        "DFlash speculative decoding does not support cache eviction.");
    if (main_model_desc.inputs_embedder) {
        m_inputs_embedder = main_model_desc.inputs_embedder;
        m_model_input_type = ModelInputType::EMBEDDINGS;
        m_embeddings_registry = std::make_shared<EmbeddingsRegistry>();
    }
    // Detect each draft capability independently from its I/O signature (no RT-info markers) and
    // apply only the missing transform(s); a well-formed draft has exactly one input/output per pair.
    const bool needs_embedding_attach = utils::has_input(draft_model, "inputs_embeds");
    const bool needs_lm_head_graft = model_has_output(draft_model, "last_hidden_state");
    OPENVINO_ASSERT(needs_embedding_attach != utils::has_input(draft_model, "input_ids"),
                    "DFlash draft model must have exactly one of 'inputs_embeds' or 'input_ids' input.");
    OPENVINO_ASSERT(needs_lm_head_graft != model_has_output(draft_model, "logits"),
                    "DFlash draft model must have exactly one of 'last_hidden_state' or 'logits' output.");

    // Resolve authoritative metadata before mutating either model so a malformed target leaves the draft reusable.
    auto retained_hidden_state_locators =
        utils::dflash::resolve_target_hidden_state_locators(main_model, m_rt_info.target_layer_ids);

    // The VLM draft keeps its external inputs_embeds; the LLM draft gets the target embedding grafted in.
    if (m_inputs_embedder) {
        OPENVINO_ASSERT(needs_embedding_attach, "DFlash VLM draft model must be inputs_embeds-native.");
    } else if (needs_embedding_attach) {
        utils::dflash::attach_target_embedding_to_draft(main_model, draft_model);
    }
    if (needs_lm_head_graft) {
        utils::dflash::attach_target_lm_head_to_draft(main_model, draft_model);
    }

    if (gguf_draft && !utils::has_input(draft_model, "token_type_ids")) {
        initialize_gguf_draft(main_model_desc, draft_model_desc, retained_hidden_state_locators);
        return;
    }

    auto [main_scheduler_config, draft_scheduler_config] = init_speculative_models(main_model_desc, draft_model_desc);
    OPENVINO_ASSERT(main_scheduler_config.enable_prefix_caching == draft_scheduler_config.enable_prefix_caching,
                    "DFlash main and draft pipelines must use the same enable_prefix_caching setting.");
    // The draft candidates are read at fixed rows of the gathered logits.
    OPENVINO_ASSERT(utils::has_input(draft_model, "sampled_tokens_indices"),
                    "DFlash draft model has no 'sampled_tokens_indices' input, so its candidate logits cannot be gathered.");
    validate_target_has_no_unmanaged_state(main_model);
    utils::dflash::expose_target_hidden_states(main_model, retained_hidden_state_locators, m_rt_info.target_layer_ids);
    utils::dflash::move_fc_from_draft_to_main(draft_model, main_model);

    // The budget never limits a draft window, so the scheduler never splits a bidirectional block. A draft
    // step is still bounded by the target's previous step plus one block per request.
    draft_scheduler_config.num_linear_attention_blocks = 0;
    draft_scheduler_config.max_num_batched_tokens = std::numeric_limits<size_t>::max();

    m_tokenizer = main_model_desc.tokenizer;
    m_generation_config = main_model_desc.generation_config;
    const std::string draft_device = draft_model_desc.device.empty() ? main_model_desc.device : draft_model_desc.device;
    const ov::AnyMap draft_properties =
        draft_model_desc.properties.empty() ? main_model_desc.properties : draft_model_desc.properties;
    std::shared_ptr<dflash_cb::CandidateSelector> candidate_selector;
    if (m_rt_info.markov_head_computation) {
        candidate_selector = std::make_shared<dflash_cb::CandidateSelector>(m_rt_info, draft_device, draft_properties);
        m_rt_info.markov_head_computation.reset();
    }

    if (m_inputs_embedder) {
        m_main_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(main_model,
                                                                                   m_inputs_embedder,
                                                                                   m_tokenizer,
                                                                                   m_generation_config,
                                                                                   main_scheduler_config,
                                                                                   main_model_desc.device,
                                                                                   main_model_desc.properties,
                                                                                   true);
        m_draft_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(draft_model,
                                                                                    m_inputs_embedder,
                                                                                    m_tokenizer,
                                                                                    m_generation_config,
                                                                                    draft_scheduler_config,
                                                                                    draft_device,
                                                                                    draft_properties,
                                                                                    false,
                                                                                    m_rt_info.mask_token_id,
                                                                                    m_rt_info.candidate_position_offset,
                                                                                    candidate_selector);
    } else {
        m_main_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(main_model,
                                                                                   m_tokenizer,
                                                                                   m_generation_config,
                                                                                   main_scheduler_config,
                                                                                   main_model_desc.device,
                                                                                   main_model_desc.properties,
                                                                                   true);
        m_draft_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(draft_model,
                                                                                    m_tokenizer,
                                                                                    m_generation_config,
                                                                                    draft_scheduler_config,
                                                                                    draft_device,
                                                                                    draft_properties,
                                                                                    false,
                                                                                    m_rt_info.mask_token_id,
                                                                                    m_rt_info.candidate_position_offset,
                                                                                    candidate_selector);
    }
    m_main_pipeline->set_hidden_state_export_needed(true);

    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
    m_draft_pipeline->raw_perf_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::initialize_gguf_draft(
    const ov::genai::ModelDesc& main_model_desc,
    const ov::genai::ModelDesc& draft_model_desc,
    const std::optional<std::vector<utils::dflash::DFlashHiddenStateLocator>>& locators) {
    OPENVINO_ASSERT(!main_model_desc.scheduler_config.enable_prefix_caching,
                    "GGUF DFlash does not support prefix caching.");
    OPENVINO_ASSERT(!m_rt_info.markov_head_computation, "GGUF DFlash does not support external Markov heads.");
    auto main_model = main_model_desc.model;
    const bool has_linear_attention = utils::get_cache_types(*main_model).has_linear();
    const bool allow_qq_bias = main_model_desc.properties.count("query_to_query_bias") > 0 &&
                               main_model_desc.properties.at("query_to_query_bias").as<bool>();
    ov::pass::SDPAToPagedAttention(false, false, true, false, false, false, allow_qq_bias).run_on_model(main_model);
    utils::apply_gather_before_matmul_transformation(main_model);
    validate_target_has_no_unmanaged_state(main_model);
    utils::dflash::expose_target_hidden_states(main_model, locators, m_rt_info.target_layer_ids);

    m_tokenizer = main_model_desc.tokenizer;
    m_generation_config = main_model_desc.generation_config;
    auto target_scheduler_config = main_model_desc.scheduler_config;
    const size_t default_assistant_tokens =
        m_generation_config.num_assistant_tokens.value_or(dflash_cb::DEFAULT_NUM_ASSISTANT_TOKENS);
    OPENVINO_ASSERT(default_assistant_tokens < target_scheduler_config.max_num_batched_tokens,
                    "GGUF DFlash requires max_num_batched_tokens >= num_assistant_tokens + 1.");
    m_gguf_max_assistant_tokens =
        dflash_cb::max_assistant_tokens_for_linear_attention(target_scheduler_config.max_num_batched_tokens - 1,
                                                             target_scheduler_config.num_linear_attention_blocks,
                                                             has_linear_attention);
    auto draft_desc = draft_model_desc;
    if (draft_desc.device.empty())
        draft_desc.device = main_model_desc.device;
    if (draft_desc.properties.empty())
        draft_desc.properties = main_model_desc.properties;
    utils::dflash::reshape_draft_hidden_states_input_for_cb(draft_desc.model);
    const auto draft_embeddings = m_inputs_embedder ? m_inputs_embedder->get_embedding_model() : nullptr;
    m_gguf_draft = std::make_shared<DFlashCBDraftRunner>(draft_desc, m_tokenizer, m_rt_info, draft_embeddings);

    if (m_inputs_embedder) {
        m_main_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(main_model,
                                                                                    m_inputs_embedder,
                                                                                    m_tokenizer,
                                                                                    m_generation_config,
                                                                                    target_scheduler_config,
                                                                                    main_model_desc.device,
                                                                                    main_model_desc.properties,
                                                                                    true);
    } else {
        m_main_pipeline = std::make_shared<ContinuousBatchingForDFlashDecodingImpl>(main_model,
                                                                                    m_tokenizer,
                                                                                    m_generation_config,
                                                                                    target_scheduler_config,
                                                                                    main_model_desc.device,
                                                                                    main_model_desc.properties,
                                                                                    true);
    }
    m_main_pipeline->set_hidden_state_export_needed(true);
    m_perf_metrics = ov::genai::SDPerModelsPerfMetrics();
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_request_config(const GenerationConfig& config) const {
    OPENVINO_ASSERT(config.is_greedy_decoding(), "DFlash supports greedy decoding only.");
    OPENVINO_ASSERT(!config.adapters.has_value(),
                    "DFlash does not support adapters until target and draft adapter parity is validated.");
    OPENVINO_ASSERT(config.assistant_confidence_threshold == 0.f,
                    "DFlash supports num_assistant_tokens only; assistant_confidence_threshold must be 0.f.");
    OPENVINO_ASSERT(config.max_ngram_size == 0, "DFlash does not support prompt lookup decoding; max_ngram_size must be 0.");
    OPENVINO_ASSERT(!config.return_omni_outputs, "DFlash does not support Omni intermediate outputs.");
}

GenerationConfig ContinuousBatchingPipeline::DFlashDecodingImpl::make_draft_generation_config(
    const GenerationConfig& config) {
    // The target decides when to stop and what to return; the draft only proposes candidates. ignore_eos
    // disables the stop tokens; they stay because validate() requires eos_token_id among them.
    GenerationConfig draft_config = config;
    draft_config.ignore_eos = true;
    draft_config.stop_strings.clear();
    draft_config.include_stop_str_in_output = false;
    draft_config.structured_output_config.reset();
    draft_config.parsers.clear();
    draft_config.echo = false;
    draft_config.logprobs = 0;
    if (draft_config.max_new_tokens == SIZE_MAX && draft_config.max_length == SIZE_MAX) {
        // ignore_eos requires a finite generation limit.
        draft_config.max_new_tokens = SIZE_MAX - 1;
    }
    return draft_config;
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::align_request_pair_processed_prefix(uint64_t request_id) {
    if (!m_main_pipeline->is_prefix_caching_enabled()) {
        return;
    }
    const auto main_group = m_main_pipeline->find_awaiting_request(request_id);
    OPENVINO_ASSERT(main_group, "DFlash prefix alignment requires the awaiting target request: ", request_id);
    // The target recomputes at least the last prompt token, which yields the first generated token.
    size_t ceiling = main_group->get_prompt_len() - 1;
    if (!m_draft_pipeline->find_awaiting_request(request_id)) {
        m_main_pipeline->restore_awaiting_prefix(request_id, ceiling);
        return;
    }
    // The draft needs target hidden states from its restored length on, so both restore the same prefix.
    while (true) {
        const size_t main_processed = m_main_pipeline->restore_awaiting_prefix(request_id, ceiling);
        const size_t draft_processed = m_draft_pipeline->restore_awaiting_prefix(request_id, main_processed);
        if (main_processed == draft_processed) {
            return;
        }
        OPENVINO_ASSERT(draft_processed < ceiling, "DFlash prefix alignment must select an earlier prefix.");
        ceiling = draft_processed;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::validate_awaiting_requests(
    const std::vector<SequenceGroup::Ptr>& main_awaiting_requests,
    const std::vector<SequenceGroup::Ptr>& draft_awaiting_requests) const {
    const size_t expected_draft_requests =
        std::count_if(main_awaiting_requests.begin(), main_awaiting_requests.end(), [](const SequenceGroup::Ptr& request) {
            return has_draft_request(request->get_sampling_parameters());
        });
    OPENVINO_ASSERT(draft_awaiting_requests.size() == expected_draft_requests,
                    "DFlash awaiting request mismatch: draft queue size is ", draft_awaiting_requests.size(),
                    ", but expected ", expected_draft_requests, " (main requests with drafting enabled).");
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const ov::genai::GenerationConfig& sampling_params,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    validate_request_config(sampling_params);
    const auto input_shape = input_ids.get_shape();
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::f32 && input_shape.size() == 3 &&
                            input_shape[0] == 1 && input_shape[1] > 0,
                        "DFlash expects input embeddings with shape [1, rows, hidden]; got ", input_shape);
    } else {
        OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::i64 && input_shape.size() == 2 &&
                            input_shape[0] == 1 && input_shape[1] > 0,
                        "DFlash expects input_ids with shape [1, seq_len]; got ", input_shape);
    }
    OPENVINO_ASSERT(m_model_input_type != ModelInputType::EMBEDDINGS || !m_main_pipeline->is_prefix_caching_enabled() ||
                        (prompt_ids.has_value() && prompt_ids->get_element_type() == ov::element::i64 &&
                         prompt_ids->get_size() == input_ids.get_shape()[1]),
                    "DFlash prefix caching requires built-in prompt preparation with one token ID per embedding");
    auto main_sampling_params = sampling_params;
    if (!main_sampling_params.num_assistant_tokens.has_value()) {
        main_sampling_params.num_assistant_tokens = m_main_pipeline->default_num_assistant_tokens;
    }

    if (m_gguf_draft) {
        return add_gguf_request(request_id, input_ids, main_sampling_params, prompt_ids, lm_extra_inputs);
    }

    // The draft prompt is the target prompt: draft position t pairs with target hidden state t.
    // insert_or_assign drops a stale handle a finished request may leave under the same request_id.
    try {
        if (has_draft_request(main_sampling_params)) {
            m_draft_generations.insert_or_assign(
                request_id,
                m_draft_pipeline->add_request(request_id, input_ids, make_draft_generation_config(main_sampling_params), prompt_ids));
        }
        auto main_handle =
            m_main_pipeline->add_request(request_id, input_ids, main_sampling_params, prompt_ids, lm_extra_inputs);
        align_request_pair_processed_prefix(request_id);
        return main_handle;
    } catch (...) {
        m_main_pipeline->discard_awaiting_request(request_id);
        m_draft_pipeline->discard_awaiting_request(request_id);
        m_draft_generations.erase(request_id);
        throw;
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_gguf_request(
    uint64_t request_id,
    const ov::Tensor& input_ids,
    const GenerationConfig& sampling_params,
    std::optional<ov::Tensor> prompt_ids,
    std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs) {
    drop_finished_gguf_requests();
    OPENVINO_ASSERT(m_request_states.empty() && !m_main_pipeline->has_non_finished_requests(),
                    "GGUF DFlash supports only one active request.");
    OPENVINO_ASSERT(sampling_params.num_assistant_tokens.value() <= m_gguf_max_assistant_tokens,
                    "GGUF DFlash num_assistant_tokens ",
                    sampling_params.num_assistant_tokens.value(),
                    " exceeds the configured scheduler capacity ",
                    m_gguf_max_assistant_tokens,
                    "; increase max_num_batched_tokens or num_linear_attention_blocks.");
    const bool is_vlm = m_model_input_type == ModelInputType::EMBEDDINGS;
    if (is_vlm)
        dflash_cb::ensure_vlm_generation_config(sampling_params);

    RequestState state;
    state.generation_config = sampling_params;
    state.prompt_len = input_ids.get_shape()[1];
    if (has_draft_request(sampling_params)) {
        auto draft_config = make_draft_generation_config(sampling_params);
        if (is_vlm) {
            draft_config.repetition_penalty = 1.0f;
            draft_config.presence_penalty = 0.0f;
            draft_config.frequency_penalty = 0.0f;
            draft_config.no_repeat_ngram_size = std::numeric_limits<size_t>::max();
            draft_config.min_new_tokens = 0;
        }
        dflash_cb::add_draft_assistant_token_headroom(draft_config, sampling_params);
        draft_config.num_assistant_tokens = 0;
        if (is_vlm)
            m_gguf_draft->initialize_sequence(state.prompt_len, draft_config);
        else
            m_gguf_draft->initialize_sequence(input_ids, draft_config);
    }
    m_request_states[request_id] = std::move(state);
    try {
        auto handle = m_main_pipeline->add_request(request_id, input_ids, sampling_params, prompt_ids, lm_extra_inputs);
        auto& request = m_request_states.at(request_id).main_request;
        request = m_main_pipeline->find_awaiting_request(request_id);
        OPENVINO_ASSERT(request, "DFlash target request was not queued: ", request_id);
        return handle;
    } catch (...) {
        m_main_pipeline->discard_awaiting_request(request_id);
        m_request_states.erase(request_id);
        throw;
    }
}

GenerationHandle ContinuousBatchingPipeline::DFlashDecodingImpl::add_request(
    uint64_t request_id,
    const std::string& prompt,
    const ov::genai::GenerationConfig& sampling_params) {
    if (m_model_input_type == ModelInputType::EMBEDDINGS) {
        return ContinuousBatchingPipeline::IContinuousBatchingPipeline::add_request(request_id,
                                                                                    prompt,
                                                                                    {},
                                                                                    sampling_params);
    }
    return add_request(request_id, m_tokenizer.encode(prompt).input_ids, sampling_params);
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step() {
    if (!m_gguf_draft) {
        SpeculativeDecodingImpl::step();
        return;
    }
    std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
    try {
        step_gguf();
    } catch (...) {
        const auto error = std::current_exception();
        m_main_pipeline->fail_pipeline(error);
        m_request_states.clear();
        std::rethrow_exception(error);
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::step_gguf() {
    auto& raw_metrics = m_perf_metrics.raw_metrics;
    auto& main_metrics = m_perf_metrics.main_model_metrics.raw_metrics;
    const auto step_start = std::chrono::steady_clock::now();
    m_main_pipeline->pull_awaiting_requests();

    std::map<uint64_t, size_t> draft_generated_by_request;
    const auto draft_start = std::chrono::steady_clock::now();
    for (auto& [request_id, state] : m_request_states) {
        if (state.finished || !has_draft_request(state.generation_config)) {
            state.pending_hidden_deltas.clear();
            state.draft_generated = 0;
            continue;
        }
        if (state.pending_hidden_deltas.empty() || state.generated_tokens.empty()) {
            state.draft_generated = 0;
            continue;
        }
        const size_t generated_len = state.generated_tokens.size();
        const size_t draft_count =
            dflash_cb::draft_candidate_count(state.generation_config.num_assistant_tokens.value(),
                                             generated_len,
                                             state.generation_config.max_new_tokens);
        const size_t validation_count =
            dflash_cb::validation_candidate_count(draft_count, generated_len, state.generation_config.max_new_tokens);
        if (validation_count == 0) {
            state.pending_hidden_deltas.clear();
            state.draft_generated = 0;
            continue;
        }
        const size_t expected = state.prompt_len + generated_len - 1;
        const size_t actual = m_gguf_draft->get_consumed_hidden_states() + state.pending_hidden_deltas.token_count();
        OPENVINO_ASSERT(actual == expected, "DFlash hidden prefix length mismatch before draft inference.");
        const auto hidden_delta = state.pending_hidden_deltas.materialize();
        const auto logits = m_gguf_draft->infer(state.generated_tokens.back(), hidden_delta, draft_count);
        state.pending_hidden_deltas.clear();
        auto candidates = m_gguf_draft->sample_candidates(logits, validation_count);
        m_perf_metrics.draft_model_metrics.raw_metrics = m_gguf_draft->get_raw_perf_metrics();
        state.generated_before_draft = generated_len;
        state.draft_generated = candidates.size();
        OPENVINO_ASSERT(!candidates.empty(),
                        "DFlash draft sampler produced no candidates for request ",
                        request_id,
                        ".");
        draft_generated_by_request[request_id] = candidates.size();
        auto candidate_tokens = state.generated_tokens;
        auto candidate_log_probs = zero_log_probs(candidate_tokens.size());
        for (const auto& candidate : candidates) {
            candidate_tokens.push_back(candidate.token_id);
            candidate_log_probs.push_back(candidate.log_prob);
        }
        GeneratedSequences candidate_sequences;
        candidate_sequences.emplace(0, GeneratedSequence(candidate_tokens, candidate_log_probs));
        m_main_pipeline->update_request(request_id, candidate_sequences, false);
    }
    const auto draft_end = std::chrono::steady_clock::now();
    m_sd_metrics.draft_duration += PerfMetrics::get_microsec(draft_end - draft_start) / 1e6;

    const auto main_start = std::chrono::steady_clock::now();
    m_main_pipeline->sync_generated_embeddings();
    m_main_pipeline->step();
    const auto main_end = std::chrono::steady_clock::now();
    const auto main_duration = PerfMetrics::get_microsec(main_end - main_start);
    m_sd_metrics.main_duration += main_duration / 1e6;

    const auto main_requests = m_main_pipeline->get_generated_requests();
    update_gguf_draft_states(main_requests);
    for (auto& [request_id, state] : m_request_states) {
        if (!main_requests.count(request_id)) {
            state.finished = true;
            const auto& sequences = state.main_request->get_sequences();
            if (sequences.size() == 1 && sequences.front()->has_finished())
                state.generated_tokens = sequences.front()->get_generated_ids();
        }
    }
    for (const auto& [request_id, draft_generated] : draft_generated_by_request) {
        const auto& state = m_request_states.at(request_id);
        const auto accounting = dflash_cb::validation_accounting(draft_generated,
                                                                 state.generated_before_draft,
                                                                 state.generated_tokens.size());
        if (!accounting.target_extended)
            continue;
        m_perf_metrics.num_draft_tokens += draft_generated;
        m_perf_metrics.num_accepted_tokens += accounting.accepted;
        m_sd_metrics.update_draft_generated_len(request_id, draft_generated);
        m_sd_metrics.update_draft_accepted_tokens(request_id, accounting.accepted);
        m_sd_metrics.update_acceptance_rate(request_id,
                                            static_cast<float>(accounting.accepted) / draft_generated * 100);
    }
    m_pipeline_metrics = m_main_pipeline->get_metrics();
    const auto step_end = std::chrono::steady_clock::now();
    const auto step_duration = PerfMetrics::get_microsec(step_end - step_start);
    const auto num_generated_tokens = m_main_pipeline->get_processed_tokens_per_iteration();
    if (num_generated_tokens > 0) {
        raw_metrics.m_token_infer_durations.emplace_back(step_duration);
        raw_metrics.m_inference_durations[0] += MicroSeconds(step_duration);
        raw_metrics.m_new_token_times.emplace_back(main_end);
        raw_metrics.m_batch_sizes.emplace_back(num_generated_tokens);
        main_metrics.m_durations.push_back(MicroSeconds(main_duration));
        main_metrics.m_inference_durations[0] += MicroSeconds(m_main_pipeline->get_metrics().inference_duration);
        main_metrics.m_batch_sizes.push_back(num_generated_tokens);
        m_sd_metrics.update_generated_len(num_generated_tokens);
    }
    if (main_requests.empty()) {
        drop_finished_gguf_requests();
        if (utils::env_setup_for_print_debug_info()) {
            m_sd_metrics.print(true);
            m_sd_metrics.clean_up();
        }
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::update_gguf_draft_states(
    const GeneratedRequests& main_generated_requests) {
    for (const auto& [request_id, sequences] : main_generated_requests) {
        auto it = m_request_states.find(request_id);
        if (it == m_request_states.end() || sequences.empty() || !has_draft_request(it->second.generation_config))
            continue;
        auto& state = it->second;
        const auto& sequence = sequences.begin()->second;
        const auto accounting = dflash_cb::validation_accounting(state.draft_generated,
                                                                 state.generated_before_draft,
                                                                 sequence.token_ids.size());
        const auto hidden_delta =
            dflash_cb::truncate_normalized_hidden_state_from_end(sequence.hidden_states, accounting.rejected);
        state.pending_hidden_deltas.append(hidden_delta, sequence.token_ids.empty());
        state.generated_tokens = sequence.token_ids;
        m_gguf_draft->sync_generated_tokens(state.generated_tokens);
        state.draft_generated = 0;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_finished_gguf_requests() {
    for (auto it = m_request_states.begin(); it != m_request_states.end();) {
        if (it->second.finished)
            it = m_request_states.erase(it);
        else
            ++it;
    }
}

void ContinuousBatchingPipeline::DFlashDecodingImpl::drop_requests() {
    if (m_gguf_draft) {
        std::lock_guard<std::mutex> lock(m_draft_generations_mutex);
        SpeculativeDecodingImpl::drop_requests();
        m_request_states.clear();
    } else {
        SpeculativeDecodingImpl::drop_requests();
    }
}

std::vector<EncodedGenerationResult> ContinuousBatchingPipeline::DFlashDecodingImpl::generate(
    const std::vector<ov::Tensor>& input_ids,
    const std::vector<GenerationConfig>& sampling_params,
    const StreamerVariant& streamer,
    const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids,
    const std::optional<std::vector<ov::Tensor>>& prompt_ids,
    const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list) {
    if (m_gguf_draft) {
        OPENVINO_ASSERT(input_ids.size() == 1, "GGUF DFlash supports batch size 1 only.");
        OPENVINO_ASSERT(!position_ids || m_model_input_type == ModelInputType::EMBEDDINGS,
                        "GGUF DFlash accepts explicit position_ids only for VLM embeddings.");
    }
    auto positions = position_ids;
    if (m_gguf_draft && positions)
        for (auto& [ids, delta] : *positions)
            if (!delta)
                delta = compute_rope_delta(ids);

    GenerateStrategy strategy;
    // The configs pass through unchanged: add_request fills an unset num_assistant_tokens with the default and
    // keeps an explicit 0, which runs the request on the target alone.
    strategy.prepare_request = [](size_t, const ov::Tensor& in_ids, GenerationConfig&, ov::Tensor& main_in) {
        main_in = in_ids;
    };
    strategy.check_streaming = [](const std::shared_ptr<ThreadedStreamerWrapper>& streamer_ptr,
                                  const std::vector<ov::Tensor>& input_ids,
                                  const std::vector<GenerationConfig>&) {
        OPENVINO_ASSERT(!streamer_ptr->has_callback() || input_ids.size() == 1,
                        "DFlash streaming only supports batch size=1.");
    };
    strategy.start_timer = []() {
        return std::chrono::steady_clock::now();
    };
    strategy.stop_timer = [](const TimePoint& start) {
        return PerfMetrics::get_microsec(std::chrono::steady_clock::now() - start);
    };

    return generate_common(this,
                           input_ids,
                           sampling_params,
                           streamer,
                           positions,
                           prompt_ids,
                           lm_extra_inputs_list,
                           strategy);
}

}  // namespace ov::genai
