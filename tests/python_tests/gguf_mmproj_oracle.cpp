// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Offline CPU reference only; never linked into GenAI.
// Usage: oracle language.gguf mmproj.gguf media-file[;media-file...]|- prompt.txt history.txt
// Reports the reference greedy choice at each step on the supplied token history.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

int main(int argc, char** argv) {
    if (argc < 6 || argc > 8)
        return 2;
    bool merge_frames = false;
    std::string encoder_output;
    for (int i = 6; i < argc; ++i) {
        if (std::string(argv[i]) == "--merge-frames")
            merge_frames = true;
        else
            encoder_output = argv[i];
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp);
    if (!model)
        return 3;
    auto cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = 4096;
    cp.n_threads = cp.n_threads_batch = 4;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* context = llama_init_from_model(model, cp);
    auto mmp = mtmd_context_params_default();
    mmp.use_gpu = false;
    mmp.warmup = false;
    mmp.n_threads = 4;
    mmp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    auto* multimodal = mtmd_init_from_file(argv[2], model, mmp);
    if (!context || !multimodal)
        return 4;
    std::ifstream text_file(argv[4]);
    std::string prompt((std::istreambuf_iterator<char>(text_file)), {});
    mtmd_input_text text{prompt.data(), prompt.size(), false, true};
    std::vector<const mtmd_bitmap*> images;
    std::vector<mtmd_bitmap*> bitmaps;
    if (std::string(argv[3]) != "-") {
        std::istringstream paths(argv[3]);
        std::string path;
        while (std::getline(paths, path, ';')) {
            auto bitmap =
                mtmd_helper_bitmap_init_from_file(multimodal, path.c_str(), false, mtmd_helper_init_opt_default());
            if (!bitmap.bitmap)
                return 5;
            mtmd_bitmap_set_mergeable(bitmap.bitmap, merge_frames);
            bitmaps.push_back(bitmap.bitmap);
            images.push_back(bitmap.bitmap);
        }
    }
    auto* chunks = mtmd_input_chunks_init();
    if (mtmd_tokenize(multimodal, chunks, &text, images.data(), images.size()))
        return 6;
    if (!encoder_output.empty()) {
        std::ofstream output(encoder_output, std::ios::binary);
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); ++i) {
            const auto* chunk = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT)
                continue;
            if (mtmd_encode_chunk(multimodal, chunk))
                return 9;
            const auto size = mtmd_input_chunk_get_n_tokens(chunk) * llama_model_n_embd_inp(model);
            output.write(reinterpret_cast<const char*>(mtmd_get_output_embd(multimodal)), size * sizeof(float));
        }
    }
    llama_pos past = 0;
    if (mtmd_helper_eval_chunks(multimodal, context, chunks, 0, 0, 4096, true, &past))
        return 7;
    std::ifstream history_file(argv[5]);
    std::vector<llama_token> history;
    llama_token token;
    while (history_file >> token)
        history.push_back(token);
    const auto vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> choices;
    auto batch = llama_batch_init(1, 0, 1);
    batch.n_tokens = 1;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = true;
    for (size_t i = 0; i < history.size(); ++i) {
        const float* logits = llama_get_logits_ith(context, -1);
        choices.push_back(std::max_element(logits, logits + vocab_size) - logits);
        if (i + 1 < history.size()) {
            batch.token[0] = history[i];
            batch.pos[0] = past++;
            if (llama_decode(context, batch))
                return 8;
        }
    }
    std::cout << "CHOICES";
    for (auto choice : choices)
        std::cout << ' ' << choice;
    std::cout << '\n';
    llama_batch_free(batch);
    mtmd_input_chunks_free(chunks);
    for (auto* bitmap : bitmaps)
        mtmd_bitmap_free(bitmap);
    mtmd_free(multimodal);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
