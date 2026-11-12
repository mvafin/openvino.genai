// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "fast_draft_strategy.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace ov::genai {

// DFlash speculative decoding on continuous batching: the target validates candidates and exports the
// hidden states of the requested layers; the draft proposes each request's candidates in one forward
// over the window [target hidden-state context rows ; seed + mask block rows].
class ContinuousBatchingPipeline::DFlashDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    template <class Impl>
    friend std::vector<EncodedGenerationResult> generate_common(
        Impl*,
        const std::vector<ov::Tensor>&,
        const std::vector<GenerationConfig>&,
        const StreamerVariant&,
        std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>,
        std::optional<std::vector<ov::Tensor>>,
        const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>&,
        GenerateStrategy&);

    DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                       const ov::genai::ModelDesc& draft_model_desc,
                       const ov::genai::utils::dflash::DFlashRTInfo& rt_info);

    GenerationHandle add_request(
        uint64_t request_id,
        const ov::Tensor& input_ids,
        const ov::genai::GenerationConfig& sampling_params,
        std::optional<ov::Tensor> prompt_ids = std::nullopt,
        std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    void step() override;

    std::vector<EncodedGenerationResult> generate(
        const std::vector<ov::Tensor>& input_ids,
        const std::vector<GenerationConfig>& sampling_params,
        const StreamerVariant& streamer,
        const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
        const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
        const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list =
            std::nullopt) override;

protected:
    DFlashDecodingImpl() = default;
    void align_request_pair_processed_prefix(uint64_t request_id) override;
    void validate_awaiting_requests(const std::vector<SequenceGroup::Ptr>& main_awaiting_requests,
                                    const std::vector<SequenceGroup::Ptr>& draft_awaiting_requests) const override;
    void validate_request_config(const GenerationConfig& config) const;
    static GenerationConfig make_draft_generation_config(const GenerationConfig& config);

    ov::genai::utils::dflash::DFlashRTInfo m_rt_info;

private:
    class DFlashCBDraftRunner;
    struct RequestState {
        dflash_cb::HiddenDeltaBuffer pending_hidden_deltas;
        std::vector<int64_t> generated_tokens;
        size_t prompt_len = 0;
        size_t generated_before_draft = 0;
        size_t draft_generated = 0;
        bool finished = false;
        GenerationConfig generation_config;
        SequenceGroup::Ptr main_request;
    };

    void initialize_gguf_draft(const ov::genai::ModelDesc& main_model_desc,
                               const ov::genai::ModelDesc& draft_model_desc,
                               const std::optional<std::vector<utils::dflash::DFlashHiddenStateLocator>>& locators);
    GenerationHandle add_gguf_request(uint64_t request_id,
                                      const ov::Tensor& input_ids,
                                      const GenerationConfig& sampling_params,
                                      std::optional<ov::Tensor> prompt_ids,
                                      std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs);
    void step_gguf();
    void update_gguf_draft_states(const GeneratedRequests& main_generated_requests);
    void drop_finished_gguf_requests();
    void drop_requests() override;

    std::shared_ptr<DFlashCBDraftRunner> m_gguf_draft;
    std::map<uint64_t, RequestState> m_request_states;
    size_t m_gguf_max_assistant_tokens = 0;
};

}  // namespace ov::genai
