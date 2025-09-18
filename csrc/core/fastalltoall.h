/*
 * This is a minimal version of https://github.com/A-Dying-Pig/fastalltoall
 * for Inter-Node only all2all communication. We do Tensor Parallel for Intra-Node.
 */

#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>
#include <cuda_runtime.h>


namespace attn {

struct fanout_nvshmem_buffer_t {
    uint8_t * send_buffer;
    uint8_t * recv_buffer;
    uint64_t * sync_signal;
    uint64_t * buffer_available_signal;
};

struct internode_transfer_params_t {
  // begin offset for all data sending to a remote rank
  uint64_t * sender_send_disp; // (rank_n,)
  // size of all data sending to a remote rank
  uint64_t * sender_transfer_sz;  // (rank_n,)
  // begin dst offset for all data sending to a remote rank
  uint64_t * sender_recv_disp; // (rank_n,)
  // size of all data received from a remote rank
  uint64_t * recver_transfer_sz; // (rank_n,)
};

int launch_alltoallv(
  uint32_t this_rank,
  uint32_t rank_n_per_node,
  uint32_t rank_n,
  struct fanout_nvshmem_buffer_t * buf,
  struct internode_transfer_params_t * inter_params,
  // This is a simplified intranode params ("node" is just myself)
  int64_t my_rank_send_offset,
  int64_t my_rank_recv_offset,
  int64_t my_rank_send_sz,
  cudaStream_t stream,
  int64_t buffer_size,
  bool separate_send_recv
);

void launch_buffer_availability_kernel(
  uint32_t this_rank,
  uint32_t rank_n_per_node,
  uint32_t rank_n,
  struct fanout_nvshmem_buffer_t * buf,
  bool is_release,
  cudaStream_t stream
);

struct FastA2aDispatchHelper {
  // Dispatch Helper for a faster all2all.
  // In this version, qkv are merged altogether to send/recv.
  int64_t _buffer_size;
  int64_t _rank;
  int64_t _local_rank;
  int64_t _world_size;

  fanout_nvshmem_buffer_t buffer;

  FastA2aDispatchHelper(
    int64_t rank, int64_t local_rank, int64_t world_size, int64_t buffer_size
  ): _rank(rank), _local_rank(local_rank), _world_size(world_size), _buffer_size(buffer_size) {
    buffer.send_buffer = (uint8_t *)nvshmem_malloc(_buffer_size);
    buffer.recv_buffer = (uint8_t *)nvshmem_malloc(_buffer_size);
    buffer.sync_signal = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * world_size);
    buffer.buffer_available_signal = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * world_size);
    cudaMemset(buffer.sync_signal, 0, sizeof(uint64_t) * world_size);
    cudaMemset(buffer.buffer_available_signal, 0, sizeof(uint64_t) * world_size);
  }

  ~FastA2aDispatchHelper() {
    nvshmem_free(buffer.send_buffer);
    nvshmem_free(buffer.recv_buffer);
    nvshmem_free(buffer.sync_signal);
    nvshmem_free(buffer.buffer_available_signal);
  }

  void update_buffer_size (int64_t target_size) {
    if (_buffer_size < target_size) {
      nvshmem_free(buffer.send_buffer);
      nvshmem_free(buffer.recv_buffer);
      buffer.send_buffer = (uint8_t *)nvshmem_malloc(target_size);
      buffer.recv_buffer = (uint8_t *)nvshmem_malloc(target_size);
      _buffer_size = target_size;
    }
  }
};

};  // namespace attn
