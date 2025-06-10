#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

template <typename T_q, typename T_kv>
__global__ void qkv_dispatch_kernel(
    T_q* query_out,
    T_kv* key_value_out,
    const T_q* query_in,
    const T_kv* key_value_in,
    const int32_t* query_dst_id,
    const int32_t* query_dst_offset,
    const int32_t* key_value_dst_id,
    const int32_t* key_value_dst_offset,
    size_t token,
    size_t hidden_q,
    size_t hidden_kv,
    uint32_t cp_degree
) {
    // --- SENDER-SIDE LOGIC ---
    
    int token_idx = blockIdx.x;
    if (token_idx < token) {
        // --- 1. Dispatch the query tensor ---
        int query_dest_rank = query_dst_id[token_idx];
        int query_dest_offset = query_dst_offset[token_idx];
        const T_q* query_src_ptr = query_in + token_idx * hidden_q;
        T_q* query_dest_ptr = query_out + query_dest_offset * hidden_q;

        if (threadIdx.x == 0) {
            nvshmem_putmem_nbi(query_dest_ptr, query_src_ptr, hidden_q * sizeof(T_q), query_dest_rank);
        }

        // --- 2. Dispatch the key_value tensor ---
        for (int i = 0; i < cp_degree; ++i) {
            int kv_idx = token_idx * cp_degree + i;
            int kv_dest_rank = key_value_dst_id[kv_idx];
            int kv_dest_offset = key_value_dst_offset[kv_idx];
            const T_kv* kv_src_ptr = key_value_in + token_idx * hidden_kv;
            T_kv* kv_dest_ptr = key_value_out + kv_dest_offset * hidden_kv;

            if (threadIdx.x == i) {
                nvshmem_putmem_nbi(kv_dest_ptr, kv_src_ptr, hidden_kv * sizeof(T_kv), kv_dest_rank);
            }
        }
    }

    // --- RECEIVER-SIDE SYNCHRONIZATION ---

    // The cooperative group sync ensures all threads in the grid reach this point
    // before any thread proceeds to the final, world-wide synchronization.
    cooperative_groups::this_grid().sync();

    // per grid barrier.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        nvshmem_quiet();
    }
}
