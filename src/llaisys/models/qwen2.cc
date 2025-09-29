#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "../../utils/types.hpp"
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <cmath>
#include <vector>

#define DEBUG 1
#define DEBUG_KV_CACHE 0
#define DEBUG_LAYER_INDEX 0

__C {
    struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        struct LlaisysQwen2Model *model = (struct LlaisysQwen2Model *) malloc (sizeof(LlaisysQwen2Model));
        if (!model || device_ids == nullptr) {
            fprintf(stderr, "Failed to allocate LlaisysQwen2Model\n");
            return nullptr;
        }
        // Initialize the model metadata
        model->meta = (struct LlaisysQwen2Meta *) malloc(sizeof(LlaisysQwen2Meta));

        model->weights = (struct LlaisysQwen2Weights *) malloc(sizeof(LlaisysQwen2Weights));
        memcpy(model->meta, meta, sizeof(LlaisysQwen2Meta));
        // Initialize device information
        model->device = device;
        model->ndevice = ndevice;
        if (ndevice > 0 && device_ids) {
            model->device_ids = (int *)malloc(sizeof(int) * ndevice);
            memcpy(model->device_ids, device_ids, sizeof(int) * ndevice);
        }

        std::cout << "\033[32m[Qwen2 Info]\033[0m\n";
        std::cout << "  🔢 Hidden size     : \033[33m" << meta->hs << "\033[0m\n";
        std::cout << "  🎯 Num heads       : \033[33m" << meta->nh << "\033[0m\n";
        std::cout << "  📐 Head dim        : \033[33m" << meta->dh << "\033[0m\n";
        std::cout << "  🔄 Num KV heads    : \033[33m" << meta->nkvh << "\033[0m\n";
        std::cout << "  🧮 MLP dim         : \033[33m" << meta->di << "\033[0m\n";
        std::cout << "  📚 Vocab size      : \033[33m" << meta->voc << "\033[0m\n";
        std::cout << "  🏗️  Num layers     : \033[33m" << meta->nlayer << "\033[0m\n";
        std::cout << "  🪢 RMSNorm eps     : \033[33m" << meta->epsilon << "\033[0m\n";
        std::cout << "  📐 RoPE theta      : \033[33m" << meta->theta << "\033[0m\n";
        std::cout << std::flush;

        // Input Embedding
        size_t shape_in_embed[2] = { meta->voc, meta->hs };
        model->weights->in_embed = tensorCreate(shape_in_embed, 2, meta->dtype, device, device_ids[0]);

        auto alloc_layer_array_2d = [&](llaisysTensor_t *&ptr, size_t dim0, size_t dim1) {
            ptr = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
            for (size_t i = 0; i < meta->nlayer; ++i) {
                size_t shape[2] = { dim0, dim1 };
                ptr[i] = tensorCreate(shape, 2, meta->dtype, model->device, model->device_ids[0]);
            }
        };

        auto alloc_layer_array_1d = [&](llaisysTensor_t *&ptr, size_t dim0) {
            ptr = (llaisysTensor_t *)malloc(sizeof(llaisysTensor_t) * meta->nlayer);
            for (size_t i = 0; i < meta->nlayer; ++i) {
                size_t shape[1] = { dim0 };
                ptr[i] = tensorCreate(shape, 1, meta->dtype, model->device, model->device_ids[0]);
            }
        };

        // Self-Attention
        alloc_layer_array_1d(model->weights->attn_norm_w, meta->hs);                     // [1536]
        alloc_layer_array_2d(model->weights->attn_q_w, meta->hs, meta->nh * meta->dh);   // [1536, 1536]
        alloc_layer_array_1d(model->weights->attn_q_b, meta->nh * meta->dh);             // [1536]
        alloc_layer_array_2d(model->weights->attn_k_w, meta->nkvh * meta->dh, meta->hs); // [256, 1536]
        alloc_layer_array_1d(model->weights->attn_k_b, meta->nkvh * meta->dh);           // [256]
        alloc_layer_array_2d(model->weights->attn_v_w, meta->nkvh * meta->dh, meta->hs); // [256, 1536]
        alloc_layer_array_1d(model->weights->attn_v_b, meta->nkvh * meta->dh);           // [256]
        alloc_layer_array_2d(model->weights->attn_o_w, meta->nh * meta->dh, meta->hs);   // [1536, 1536]

        // MLP
        alloc_layer_array_1d(model->weights->mlp_norm_w, meta->hs);             // [1536]
        alloc_layer_array_2d(model->weights->mlp_gate_w, meta->di, meta->hs);   // [8960, 1536]
        alloc_layer_array_2d(model->weights->mlp_up_w, meta->di, meta->hs);     // [8960, 1536]
        alloc_layer_array_2d(model->weights->mlp_down_w, meta->hs, meta->di);   // [1536, 8960]

        // Output Layer Norm
        size_t shape_out_norm[1] = { meta->hs };
        model->weights->out_norm_w = tensorCreate(shape_out_norm, 1, meta->dtype, model->device, model->device_ids[0]);

        // Output Embedding
        size_t shape_out_embed[2] = { meta->voc, meta->hs };
        model->weights->out_embed = tensorCreate(shape_out_embed, 2, meta->dtype, device, device_ids[0]);

        return model;
    }

    void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        if (!model) return;

        tensorDestroy(model->weights->in_embed);
        tensorDestroy(model->weights->out_embed);
        tensorDestroy(model->weights->out_norm_w);

        // attn_norm_w (1d tensor array)
        if (model->weights->attn_norm_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_norm_w[i]);
            }
            free(model->weights->attn_norm_w);
        }

        // attn_q_w (2d tensor array)
        if (model->weights->attn_q_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_q_w[i]);
            }
            free(model->weights->attn_q_w);
        }

        // attn_q_b (1d tensor array)
        if (model->weights->attn_q_b) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_q_b[i]);
            }
            free(model->weights->attn_q_b);
        }

        // attn_k_w (2d tensor array)
        if (model->weights->attn_k_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_k_w[i]);
            }
            free(model->weights->attn_k_w);
        }

        // attn_k_b (1d tensor array)
        if (model->weights->attn_k_b) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_k_b[i]);
            }
            free(model->weights->attn_k_b);
        }

        // attn_v_w (2d tensor array)
        if (model->weights->attn_v_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_v_w[i]);
            }
            free(model->weights->attn_v_w);
        }

        // attn_v_b (1d tensor array)
        if (model->weights->attn_v_b) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_v_b[i]);
            }
            free(model->weights->attn_v_b);
        }

        // attn_o_w (2d tensor array)
        if (model->weights->attn_o_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->attn_o_w[i]);
            }
            free(model->weights->attn_o_w);
        }

        // mlp_norm_w (1d tensor array)
        if (model->weights->mlp_norm_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_norm_w[i]);
            }
            free(model->weights->mlp_norm_w);
        }

        // mlp_gate_w (2d tensor array)
        if (model->weights->mlp_gate_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_gate_w[i]);
            }
            free(model->weights->mlp_gate_w);
        }

        // mlp_up_w (2d tensor array)
        if (model->weights->mlp_up_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_up_w[i]);
            }
            free(model->weights->mlp_up_w);
        }

        // mlp_down_w (2d tensor array)
        if (model->weights->mlp_down_w) {
            for (size_t i = 0; i < model->meta->nlayer; ++i) {
                tensorDestroy(model->weights->mlp_down_w[i]);
            }
            free(model->weights->mlp_down_w);
        }

        if (model->device_ids) {
            free(model->device_ids);
        }

        if (model->meta) {
            free(model->meta);
        }

        free(model);
    }

    struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return model->weights;
    }

    int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t * token_ids, size_t ntoken, llaisysTensor_t *kcache, llaisysTensor_t *vcache, size_t past_len) {
        if (!model || !token_ids || ntoken == 0) return -1;

        // If kv_cache != nullptr, it means KV Cache is used for performance.
        bool kv_cache_used = (kcache != nullptr && vcache != nullptr);
        // kcache [max_seq, nkvh, d], vcache [max_seq, nkvh, dv]

        size_t seqlen = ntoken;
        size_t hs = model->meta->hs;       // hidden size
        size_t nh = model->meta->nh;       // num heads
        size_t dh = model->meta->dh;       // head dim
        size_t nkvh = model->meta->nkvh;   // num key-value heads
        size_t di = model->meta->di;       // mlp intermediate dim
        size_t voc = model->meta->voc;     // vocab size
        size_t nlayer = model->meta->nlayer;
        float scale = 1.0f / std::sqrt(static_cast<float>(dh)); // scale for attention
        float rms_eps = model->meta->epsilon; // epsilon for RMSNorm
        float rope_theta = model->meta->theta; // theta for RoPE

        // 1. intput token_ids -> tensor
        size_t input_tensor_shape[1] = {seqlen};
        llaisysTensor_t input_tensor = tensorCreate(input_tensor_shape, 1, LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        tensorLoad(input_tensor, token_ids);
        if(DEBUG){
            std::cout << std::endl << "input_tensor:" << std::endl;    
            tensorDebug(input_tensor); // Debugging: print first token's attention output
        }
        // 2. Embedding lookup: [seqlen] -> [seqlen, hs]
        size_t output_embedding_tensor_shape[2] = {seqlen, hs};
        llaisysTensor_t output_embedding_tensor = tensorCreate(output_embedding_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysEmbedding(output_embedding_tensor, input_tensor, model->weights->in_embed);        
        if(DEBUG){
            std::cout << std::endl << "output_embedding_tensor:" << std::endl;    
            tensorDebug(tensorSlice(output_embedding_tensor, 0, 0,1)); // Debugging: print first token's attention output
        }
        /******************************************************************************************
         *  Embedding Layer has been checked. 
         *******************************************************************************************/
        
        // output_hidden_layer_tensor is used to store the output of the hidden layer
        llaisysTensor_t output_hidden_layer_tensor = output_embedding_tensor;
        size_t position_shape[1] = {seqlen};
        llaisysTensor_t position_ids = tensorCreate(position_shape, 1, LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        // int64_t* pos_data = (int64_t*)tensorGetData(position_ids);
        std::vector<int64_t> host_pos(seqlen);
        for (size_t i = 0; i < seqlen; i++) {
            if(kv_cache_used)
                host_pos[i] = (int64_t)(past_len + i);  // When using KV cache, position ids continue from past_len
            else
                host_pos[i] = (int64_t) i;
        }
        tensorLoad(position_ids, host_pos.data());

        if(DEBUG) {
            std::cout << std::endl << "position_ids:" << std::endl; 
            tensorDebug(position_ids); // Debugging: print first token's attention output
        }

        // 3. Transformer hidden layers
        for (size_t i = 0; i < nlayer; i++) {
            // 3.1 LayerNorm before Self-attention
            llaisysTensor_t attn_norm_w = model->weights->attn_norm_w[i];
            size_t output_input_layernorm_shape[2] = {seqlen, hs};
            llaisysTensor_t output_input_layernorm_tensor = tensorCreate(output_input_layernorm_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysRmsNorm(output_input_layernorm_tensor, output_hidden_layer_tensor, attn_norm_w, rms_eps);
            if(DEBUG && i==DEBUG_LAYER_INDEX) {
                std::cout << std::endl << "output_input_layernorm_tensor:" << std::endl;    
                tensorDebug(tensorSlice(output_input_layernorm_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }
            /******************************************************************************************
             *  Rms Norm Layer has been checked. 
             *******************************************************************************************/

            // 3.2 Q projection
            llaisysTensor_t attn_q_w = model->weights->attn_q_w[i];
            llaisysTensor_t attn_q_b = model->weights->attn_q_b[i];
            size_t q_tensor_shape[2] = {seqlen, nh * dh};
            llaisysTensor_t q_tensor = tensorCreate(q_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(q_tensor, output_input_layernorm_tensor, attn_q_w, attn_q_b);
            if(DEBUG && i==DEBUG_LAYER_INDEX) {
                std::cout << std::endl << "q_tensor:" << std::endl;    
                tensorDebug(tensorSlice(q_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }
            size_t q_tensor_reshape_shape[3] = {seqlen, nh, dh};
            q_tensor = tensorReshape(q_tensor, q_tensor_reshape_shape, 3);
            llaisysTensor_t q_rope_tensor = tensorCreate(q_tensor_reshape_shape, 3, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysROPE(q_rope_tensor, q_tensor, position_ids, rope_theta);
            if(DEBUG && i==DEBUG_LAYER_INDEX) {
                std::cout << std::endl << "q_rope_tensor:" << std::endl;    
                tensorDebug(tensorSlice(q_rope_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }
            /******************************************************************************************
             *  Q Proj Layer has been checked. 
             *******************************************************************************************/

            // 3.3 K projection
            llaisysTensor_t attn_k_w = model->weights->attn_k_w[i];
            llaisysTensor_t attn_k_b = model->weights->attn_k_b[i];
            size_t k_tensor_shape[2] = {seqlen, nkvh * dh};
            llaisysTensor_t k_tensor = tensorCreate(k_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(k_tensor, output_input_layernorm_tensor, attn_k_w, attn_k_b);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "k_tensor:" << std::endl;    
                tensorDebug(tensorSlice(k_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }
            size_t k_tensor_reshape_shape[3] = {seqlen, nkvh, dh};
            k_tensor = tensorReshape(k_tensor, k_tensor_reshape_shape, 3);
            llaisysTensor_t k_rope_tensor; 
            if (kv_cache_used) {
                // When using KV cache, we need to update the kcache tensor
                // kcache shape: [max_seq, nkvh, dh]
                // Slice the kcache to get the current position to update
                k_rope_tensor = tensorSlice(kcache[i], 0, past_len, past_len + seqlen); // Write to kcache
            } else {
                k_rope_tensor = tensorCreate(k_tensor_reshape_shape, 3, model->meta->dtype, model->device, model->device_ids[0]);
            }
            llaisysROPE(k_rope_tensor, k_tensor, position_ids, rope_theta);
            if(DEBUG && i==DEBUG_LAYER_INDEX) {
                std::cout << std::endl << "k_rope_tensor:" << std::endl;    
                tensorDebug(tensorSlice(k_rope_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }
            /******************************************************************************************
             *  K Proj Layer has been checked. 
             *******************************************************************************************/

            // 3.4 V projection
            llaisysTensor_t attn_v_w = model->weights->attn_v_w[i];
            llaisysTensor_t attn_v_b = model->weights->attn_v_b[i];
            size_t v_tensor_shape[2] = {seqlen, nkvh * dh};
            llaisysTensor_t v_tensor;
            if (kv_cache_used) {
                // When using KV cache, we need to update the vcache tensor
                // vcache shape: [max_seq, nkvh, dh]
                // Slice the vcache to get the current position to update
                v_tensor = tensorView(tensorSlice(vcache[i], 0, past_len, past_len + seqlen), v_tensor_shape, 2);
            } else {
                v_tensor = tensorCreate(v_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            }
            llaisysLinear(v_tensor, output_input_layernorm_tensor, attn_v_w, attn_v_b);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "v_tensor:" << std::endl;
                tensorDebug(tensorSlice(v_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }

            // 3.5 Self-attention
            size_t output_self_attn_multihead_tensor_shape[3] = {seqlen, nh, dh};
            llaisysTensor_t output_self_attn_tensor = tensorCreate(output_self_attn_multihead_tensor_shape, 3, model->meta->dtype, model->device, model->device_ids[0]);
            
            if (kv_cache_used) {
                // Use KV cache to speed up inference
                // kcache [max_seq, nkvh, d], vcache [max_seq, nkvh, dv]
                k_rope_tensor = tensorSlice(kcache[i], 0, 0, past_len + seqlen);
                v_tensor = tensorSlice(vcache[i], 0, 0, past_len + seqlen);
                llaisysSelfAttention(output_self_attn_tensor, q_rope_tensor, k_rope_tensor, v_tensor, scale);
            } else {
                size_t v_tensor_reshape_shape[3] = {seqlen, nkvh, dh};
                v_tensor = tensorReshape(v_tensor, v_tensor_reshape_shape, 3);
                /******************************************************************************************
                 *  V Proj Layer has been checked. 
                *******************************************************************************************/
                llaisysSelfAttention(output_self_attn_tensor, q_rope_tensor, k_rope_tensor, v_tensor, scale);
            }
            size_t output_self_attn_tensor_shape[2] = {seqlen, nh * dh};
            output_self_attn_tensor = tensorReshape(output_self_attn_tensor, output_self_attn_tensor_shape, 2);
            if (DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "output_self_attn_tensor:" << std::endl;
                tensorDebug(output_self_attn_tensor); // Debugging: print first token's attention output
            }

            // 3.6 Self-attention output projection
            llaisysTensor_t attn_o_w = model->weights->attn_o_w[i];
            size_t o_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t o_tensor = tensorCreate(o_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(o_tensor, output_self_attn_tensor, attn_o_w, nullptr);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "o_tensor:" << std::endl;
                tensorDebug(tensorSlice(o_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }
            /******************************************************************************************
             *  O Proj Layer has been checked. 
             *******************************************************************************************/

            // 3.7 Residual connection after attn
            size_t output_res1_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t output_res1_tensor = tensorCreate(output_res1_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysAdd(output_res1_tensor, output_hidden_layer_tensor, o_tensor);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "output_res1_tensor:" << std::endl;
                tensorDebug(tensorSlice(output_res1_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }

            // 3.8 Post-attention LayerNorm
            llaisysTensor_t post_attn_norm_w = model->weights->mlp_norm_w[i];
            size_t output_post_self_attn_layernorm_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t output_post_self_attn_layernorm_tensor = tensorCreate(output_post_self_attn_layernorm_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysRmsNorm(output_post_self_attn_layernorm_tensor, output_res1_tensor, post_attn_norm_w, rms_eps);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "output_post_self_attn_layernorm_tensor:" << std::endl;
                tensorDebug(tensorSlice(output_post_self_attn_layernorm_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }

            // 3.9 MLP (Gate, Up, Down)
            llaisysTensor_t mlp_gate_w = model->weights->mlp_gate_w[i];
            llaisysTensor_t mlp_up_w = model->weights->mlp_up_w[i];
            llaisysTensor_t mlp_down_w = model->weights->mlp_down_w[i];

            size_t mlp_gate_tensor_shape[2] = {seqlen, di};
            llaisysTensor_t mlp_gate_tensor = tensorCreate(mlp_gate_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_gate_tensor, output_post_self_attn_layernorm_tensor, mlp_gate_w, nullptr);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "mlp_gate_tensor:" << std::endl;
                tensorDebug(tensorSlice(mlp_gate_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }

            size_t mlp_up_tensor_shape[2] = {seqlen, di};
            llaisysTensor_t mlp_up_tensor = tensorCreate(mlp_up_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_up_tensor, output_post_self_attn_layernorm_tensor, mlp_up_w, nullptr);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "mlp_up_tensor:" << std::endl;
                tensorDebug(tensorSlice(mlp_up_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }

            size_t swiglu_tensor_shape[2] = {seqlen, di};
            llaisysTensor_t swiglu_tensor = tensorCreate(swiglu_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysSwiGLU(swiglu_tensor, mlp_gate_tensor, mlp_up_tensor);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "swiglu_tensor:" << std::endl;
                tensorDebug(tensorSlice(swiglu_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }

            size_t mlp_down_tensor_shape[2] = {seqlen, hs};
            llaisysTensor_t mlp_down_tensor = tensorCreate(mlp_down_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
            llaisysLinear(mlp_down_tensor, swiglu_tensor, mlp_down_w, nullptr);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "mlp_down_tensor:" << std::endl;
                tensorDebug(tensorSlice(mlp_down_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }
            /******************************************************************************************
             * MLP Layer has been checked. 
             *******************************************************************************************/

            // 3.10 Residual connection after MLP
            llaisysAdd(output_hidden_layer_tensor, output_res1_tensor, mlp_down_tensor);
            if(DEBUG && i==DEBUG_LAYER_INDEX){
                std::cout << std::endl << "output_hidden_layer_tensor:" << std::endl;
                tensorDebug(tensorSlice(output_hidden_layer_tensor, 0, 0,1)); // Debugging: print first token's attention output
            }

            /**********************************************************************************************
             *  Check KV Cache
             **********************************************************************************************/
            if(DEBUG_KV_CACHE && i==DEBUG_LAYER_INDEX){
                tensorDebug(k_rope_tensor);
                tensorDebug(v_tensor);
            }

            if(!kv_cache_used){
                tensorDestroy(k_rope_tensor);
                tensorDestroy(v_tensor);
            }

            // release intermediate tensors
            tensorDestroy(output_input_layernorm_tensor);
            tensorDestroy(q_tensor);
            tensorDestroy(q_rope_tensor);
            tensorDestroy(k_tensor);
            tensorDestroy(output_self_attn_tensor);
            tensorDestroy(o_tensor);
            tensorDestroy(output_res1_tensor);
            tensorDestroy(output_post_self_attn_layernorm_tensor);
            tensorDestroy(mlp_gate_tensor);
            tensorDestroy(mlp_up_tensor);
            tensorDestroy(swiglu_tensor);
            tensorDestroy(mlp_down_tensor);
        }

        // 4. Output LayerNorm
        llaisysTensor_t final_layernorm_w = model->weights->out_norm_w;
        size_t output_final_layernorm_shape[2] = {seqlen, hs};
        llaisysTensor_t output_final_layernorm_tensor = tensorCreate(output_final_layernorm_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysRmsNorm(output_final_layernorm_tensor, output_hidden_layer_tensor, final_layernorm_w, rms_eps);
        if(DEBUG){
            std::cout << std::endl << "output_final_layernorm_tensor:" << std::endl;
            tensorDebug(tensorSlice(output_final_layernorm_tensor, 0, 0,1)); // Debugging: print first token's attention output
        }
        /******************************************************************************************
         * Final Layernorm Layer has been checked. 
         *******************************************************************************************/

        // 5. Output [seqlen, voc]
        size_t output_tensor_shape[2] = {seqlen, voc};
        llaisysTensor_t output_tensor = tensorCreate(output_tensor_shape, 2, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysLinear(output_tensor, output_final_layernorm_tensor, model->weights->out_embed, nullptr);
    
        /******************************************************************************************
         * Output Embedding Layer has been checked. 
         *******************************************************************************************/

        // [seqlen, voc] -> [1, voc]
        size_t output_tensor_slice_reshape_shape[1] = {voc};
        llaisysTensor_t output_tensor_slice = tensorSlice(output_tensor, 0, seqlen-1, seqlen); // only need the last token's logits
        if(DEBUG){
            std::cout << std::endl << "output_tensor_slice:" << std::endl;
            tensorDebug(tensorSlice(tensorView(output_tensor_slice, output_tensor_slice_reshape_shape, 1), 0, 0, 100)); // Debugging: print first token's attention output
        }

        size_t index_shape[1] = {1};
        llaisysTensor_t index_tensor = tensorCreate(index_shape, 1, LLAISYS_DTYPE_I64, model->device, model->device_ids[0]);
        llaisysTensor_t value_tensor = tensorCreate(index_shape, 1, model->meta->dtype, model->device, model->device_ids[0]);
        llaisysArgmax(index_tensor, value_tensor, tensorView(output_tensor_slice, output_tensor_slice_reshape_shape, 1));
        /******************************************************************************************
         * Argmax has been checked. 
         *******************************************************************************************/

        index_tensor = tensorTo(index_tensor, LLAISYS_DEVICE_CPU, 0);
        int64_t index = *((int64_t *) tensorGetData(index_tensor));

        tensorDestroy(position_ids);
        tensorDestroy(input_tensor);
        tensorDestroy(output_embedding_tensor);
        tensorDestroy(output_final_layernorm_tensor);
        tensorDestroy(output_tensor);
        tensorDestroy(index_tensor);
        tensorDestroy(value_tensor);

        return index;
    }
}   
