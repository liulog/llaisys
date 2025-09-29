#include "self_attention_nv.cuh"

#include "../../../utils.hpp"

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Note:
//      flashAttnKernel implement is from my repository https://github.com/liulog/Learning-CUDA.
//      Its implement refereces: https://github.com/tspeterkim/flash-attention-minimal.
// 
// Here I update it to support bfloat16 and half data types.
template <typename T>
__global__ void flashAttnKernel(
    const T *d_q, // [batch_size, target_seq_len, query_heads, head_dim]
    const T *d_k, // [batch_size, src_seq_len, kv_heads, head_dim]
    const T *d_v, // [batch_size, src_seq_len, kv_heads, head_dim]
    T *d_o,       // [batch_size, target_seq_len, query_heads, head_dim]
    const int Tr, const int Tc, const int Br, const int Bc,
    const int batch_size, const int target_seq_len, const int src_seq_len,
    const int query_heads, const int kv_heads, const int head_dim,
    const int group_size, const bool is_causal, const float softmax_scale,
    float *d_l, float *d_m)
{
    extern __shared__ float smem[]; // dynamic shared memory
    int q_tile_size = Br * head_dim;
    int k_tile_size = Bc * head_dim;
    int v_tile_size = Bc * head_dim;

    float *Qi = smem;             // Q [Br, head_dim]
    float *Kj = Qi + q_tile_size; // K [Bc, head_dim]
    float *Vj = Kj + k_tile_size; // V [Bc, head_dim]
    float *S = Vj + v_tile_size;  // S [Br, Bc]

    /********************************************************
     *  Check Shared Memory Addr
     ********************************************************/

    int tx = threadIdx.x; // thread within block
    int bx = blockIdx.x;  // batch index
    int by = blockIdx.y;  // query head index
    // int Br = blockDim.x

    // Compute Q/K/V/O/l/m offsets
    int q_offset = bx * target_seq_len * query_heads * head_dim + by * head_dim;          // [batch_size, target_seq_len, query_heads, head_dim]
    int k_offset = bx * src_seq_len * kv_heads * head_dim + (by / group_size) * head_dim; // [batch_size, src_seq_len, kv_heads, head_dim]
    int v_offset = k_offset;                                                              // [batch_size, src_seq_len, kv_heads, head_dim]
    int o_offset = q_offset;                                                              // [batch_size, target_seq_len, query_heads, head_dim]
    int lm_offset = bx * target_seq_len * query_heads + by;                               // [batch_size, target_seq_len, query_heads]

    for (int j = 0; j < Tc; ++j) // Outer loop
    {
        // 1. Load K and V into shared memory
        if (tx < Bc)
        {
            int k_row = j * Bc + tx;
            // Per thread load one row of K and V
            for (int x = 0; x < head_dim; ++x)
            {
                float kval = 0.f, vval = 0.f;
                if (k_row < src_seq_len)
                {
                    kval = static_cast<float>(d_k[k_offset + k_row * kv_heads * head_dim + x]); // [batch_size, src_seq_len, kv_heads, head_dim]
                    vval = static_cast<float>(d_v[v_offset + k_row * kv_heads * head_dim + x]); // [batch_size, src_seq_len, kv_heads, head_dim]
                }
                Kj[(tx * head_dim + x)] = kval; // [batch_size, src_seq_len, kv_heads, head_dim]
                Vj[(tx * head_dim + x)] = vval; // [batch_size, src_seq_len, kv_heads, head_dim]
            }
        }
        __syncthreads(); // Ensure all threads have loaded Kj and Vj

        /********************************************************
         *  Check Load Kj and Vj to Shared Memory. Ok!
         ********************************************************/

        for (int i = 0; i < Tr; ++i) // Inner loop
        {
            int q_row = i * Br + tx; // global index
            bool q_valid = (q_row < target_seq_len);

            // 2. Load Q into shared memory
            for (int x = 0; x < head_dim; ++x) // Q [Br, d]
            {
                Qi[(tx * head_dim + x)] = q_valid ? static_cast<float>(d_q[q_offset + q_row * query_heads * head_dim + x]) : 0.f; // [batch_size, target_seq_len, query_heads, head_dim]
            }

            // Initially, row_m_prev = -INFINITY, row_l_prev = 0
            float row_m_prev = -INFINITY;
            float row_l_prev = 0.f;
            if (j != 0 && q_valid)
            {
                row_m_prev = d_m[lm_offset + q_row * query_heads]; // [batch_size, target_seq_len, query_heads]
                row_l_prev = d_l[lm_offset + q_row * query_heads];
            }

            /********************************************************
             *  Check Load Qi to Shared Memory. Ok!
             ********************************************************/

            // 3. S = Q * K^T * softmax_scale, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; ++y) // [Br, d] * [d, Bc] -> [Br, Bc]
            {
                int k_pos = j * Bc + y;
                float sum = 0.f;
                for (int x = 0; x < head_dim; ++x)
                {
                    sum += Qi[tx * head_dim + x] * Kj[y * head_dim + x]; // Q * K^T
                }
                sum *= softmax_scale;
                bool valid = q_valid && (k_pos < src_seq_len);
                if (!valid) // Bound check
                    sum = -INFINITY;

                /********************************************************************************
                 * Note: I think this condition has error !!!
                 * it should be k_pos > q_row + src_seq_len - target_seq_len, considering kvcache
                 ********************************************************************************/
                if (valid && is_causal && (k_pos > q_row + src_seq_len - target_seq_len)) // k_pos > q_row + src_seq_len - target_seq_len
                    sum = -INFINITY;

                S[tx * Bc + y] = sum; // [Br, Bc]
                row_m = max(row_m, sum);
            }
            /********************************************************
             *  Check Compute S = Q * K^T, Record row_m
             ********************************************************/

            // 4. P = exp(S - row_m), row_l = row_sum(P)
            float row_l = 0.f;
            for (int y = 0; y < Bc; ++y)
            {
                // If (-inf) - (-inf), then expf will generate Nan.
                // For example, for casual attention, top right triangle part is all -inf.
                if (S[tx * Bc + y] == -INFINITY)
                    S[tx * Bc + y] = 0.f;
                else
                    S[tx * Bc + y] = expf(S[tx * Bc + y] - row_m);  // Note: here maybe generate Nan, due to （-inf) - (-inf)
                row_l += S[tx * Bc + y];
            }
            /********************************************************
             *  Check Tile Exp, Calculate row_l
             ********************************************************/

            // 5. Compute new m and l
            float row_m_new = max(row_m, row_m_prev);
            float row_l_new = (expf(row_m_prev - row_m_new) * row_l_prev) + expf(row_m - row_m_new) * row_l;

            // 6. Write O, l, m to HBM
            for (int x = 0; x < head_dim; ++x) // P * V
            {
                float pv = 0.f;
                for (int y = 0; y < Bc; ++y) // [Br, Bc] * [Bc, d] -> [Br, d]
                {
                    pv += S[tx * Bc + y] * Vj[y * head_dim + x];
                }
                // Update output
                if (q_valid)
                    d_o[o_offset + q_row * query_heads * head_dim + x] = static_cast<T>((1 / row_l_new) * ((row_l_prev * expf(row_m_prev - row_m_new) * static_cast<float>(d_o[o_offset + q_row * query_heads * head_dim + x])) + (expf(row_m - row_m_new) * pv)));
            }
            if (q_valid)
            {
                d_m[lm_offset + q_row * query_heads] = row_m_new;
                d_l[lm_offset + q_row * query_heads] = row_l_new;
            }
            /********************************************************
             *  Check Calc O, l, m
             ********************************************************/
        }
        __syncthreads();
    }
}

// Here I don't implement the self_attention_kernel, instead I call flashAttnKernel directly.
// template <typename T>
// __global__ void self_attention_kernel(T *attn_val, const T *q, const T *k, const T * v, const float scale, size_t seqlen, size_t nhead, size_t dv, size_t d, size_t total_len, size_t nkvhead) {
// }

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, std::byte *q, std::byte *k, std::byte *v, float scale, llaisysDataType_t type, size_t seqlen, size_t nhead, size_t dv, size_t d, size_t total_len, size_t nkvhead) {
    ASSERT(dv==d, "Currently only support dv == d");
    
    int head_dim = d, src_seq_len = total_len, target_seq_len = seqlen;
    int query_heads = nhead, kv_heads = nkvhead;
    int batch_size = 1;
    // std::cout << "FlashAttention Config: batch_size=" << batch_size << ", target_seq_len=" << target_seq_len << ", src_seq_len=" << src_seq_len
    //           << ", query_heads=" << query_heads << ", kv_heads=" << kv_heads << ", head_dim=" << head_dim << std::endl;

    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0);

    int Bc = 256, Br = 256;
    int smem_bytes;
    smem_bytes = (Br * head_dim + 2 * Bc * head_dim + Br * Bc) * sizeof(float);
    while (smem_bytes >= max_smem)
    {
        Br -= 16;
        Bc -= 16;
        smem_bytes = (Br * head_dim + 2 * Bc * head_dim + Br * Bc) * sizeof(float);
    }
    if (Br == 0 || Bc == 0)
    {
        throw std::runtime_error("Shared memory isn't enough for this configuration.");
    }

    const int Tc = std::ceil((float)(src_seq_len) / Bc);
    const int Tr = std::ceil((float)(target_seq_len) / Br);
    int group_size = query_heads / kv_heads;
    // std::cout << "FlashAttention Params: Br=" << Br << ", Bc=" << Bc << ", Tr=" << Tr << ", Tc=" << Tc << ", smem_bytes=" << smem_bytes / 1024.0f << "KB" << std::endl;

    dim3 grid(batch_size, query_heads);
    dim3 block(Br);

    float *d_m, *d_l;
    // Note: d_m and d_l should be used carefully.
    cudaMalloc(&d_m, sizeof(float) * batch_size * target_seq_len * query_heads);
    cudaMalloc(&d_l, sizeof(float) * batch_size * target_seq_len * query_heads);
    
    // Note: initializing attn_val to zero is very important.
    //      Because in flashAttnKernel, attn_val's old value is used in the calculation of new attn_val.
    size_t elem_size = 0;
    switch(type) {
        case LLAISYS_DTYPE_F32: elem_size = sizeof(float); break;
        case LLAISYS_DTYPE_BF16: elem_size = sizeof(__nv_bfloat16); break;
        case LLAISYS_DTYPE_F16: elem_size = sizeof(__nv_half); break;
        default: EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
    cudaMemset(attn_val, 0, elem_size * batch_size * target_seq_len * query_heads * head_dim);

    switch (type) {
    case LLAISYS_DTYPE_F32: {
        flashAttnKernel<<<grid, block, smem_bytes>>>(
            reinterpret_cast<const float*>(q),
            reinterpret_cast<const float*>(k),
            reinterpret_cast<const float*>(v),
            reinterpret_cast<float*>(attn_val),
            Tr, Tc, Br, Bc,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            group_size, true, scale,
            d_m, d_l);
        break;
    }
    case LLAISYS_DTYPE_BF16: {
        flashAttnKernel<<<grid, block, smem_bytes>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            reinterpret_cast<const __nv_bfloat16*>(k),
            reinterpret_cast<const __nv_bfloat16*>(v),
            reinterpret_cast<__nv_bfloat16*>(attn_val),
            Tr, Tc, Br, Bc,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            group_size, true, scale,
            d_m, d_l);
        break;
    }
    case LLAISYS_DTYPE_F16: {
        flashAttnKernel<<<grid, block, smem_bytes>>>(
            reinterpret_cast<const __nv_half*>(q),
            reinterpret_cast<const __nv_half*>(k),
            reinterpret_cast<const __nv_half*>(v),
            reinterpret_cast<__nv_half*>(attn_val),
            Tr, Tc, Br, Bc,
            batch_size, target_seq_len, src_seq_len,
            query_heads, kv_heads, head_dim,
            group_size, true, scale,
            d_m, d_l);
        break;
    }
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }

    cudaFree(d_m);
    cudaFree(d_l);

}
} // namespace llaisys::ops::nvidia
