#include <cooperative_groups.h>
#include <cuda.h>
#include <nvshmem.h>

#include "core/cuda_utils.h"
#include "core/fastalltoall.h"

#define THREAD_N_PER_WARP 32
#define THREAD_N_PER_2WARP 64

namespace attn {
namespace {
__global__ void spreadout_alltoallv_internode_kernel(
  // rank information
  const uint32_t this_rank,
  const uint32_t local_rank_n,
  const uint32_t rank_n,
  // nvshmem memory for RDMA data exchange
  uint8_t * send_buffer,
  uint8_t * recv_buffer,
  uint64_t * sync_signal,  
  // metadata for internode transfer
  const uint64_t * inter_sender_send_disp,
  const uint64_t * inter_sender_transfer_sz,
  const uint64_t * inter_sender_recv_disp,
  const uint64_t * inter_recver_transfer_sz
) {
  const uint32_t warp_id = threadIdx.x / THREAD_N_PER_WARP;
  const uint32_t lane_id = threadIdx.x % THREAD_N_PER_WARP;
  
  const uint32_t server_id = this_rank / local_rank_n;
  const uint32_t local_rank_id = this_rank % local_rank_n;
  const uint32_t server_n = rank_n / local_rank_n;
  const uint32_t inter_node_rank_n = rank_n - local_rank_n;

  if (warp_id == 0){
    //use warp 0 in block 0 to do inter-node transfer
    for (uint step = 1; step < server_n; step ++){
      const uint32_t dst_server_id = (server_id + step) % server_n;
      // const uint32_t src_server_id = (server_id + server_n - step) % server_n;
      for (uint j = 0; j < local_rank_n; j ++){
        const uint32_t send_rank_id = dst_server_id * local_rank_n + (local_rank_id + j) % local_rank_n;
        // const uint32_t recv_rank_id = src_server_id * local_rank_n + (local_rank_id + local_rank_n - j) % local_rank_n;
        const uint64_t send_offset = __ldg(&inter_sender_send_disp[send_rank_id]);
        const uint64_t recv_offset = __ldg(&inter_sender_recv_disp[send_rank_id]);
        const int64_t send_sz = __ldg(&inter_sender_transfer_sz[send_rank_id]);
        nvshmemx_putmem_signal_nbi_warp(
          recv_buffer + recv_offset,
          send_buffer + send_offset,
          send_sz,
          &sync_signal[this_rank],
          send_sz,
          NVSHMEM_SIGNAL_ADD,
          send_rank_id
        );
      }
    }
    for (uint i = lane_id; i < inter_node_rank_n; i += THREAD_N_PER_WARP){
      const uint32_t send_rank_id = (
        (server_id + 1) * local_rank_n + i
      ) % rank_n;
      // Send one more byte on the signal here, to guarantee that
      // there is always a communication between each peers.
      // This is to guarantee that two ranks are running the
      // communication at all times, or at most with 1 communication
      // different (so that we can use dual buffers).
      if (threadIdx.x == 0) {
        nvshmemx_signal_op(&sync_signal[this_rank], 1, NVSHMEM_SIGNAL_ADD, send_rank_id);
      }
    }
    nvshmem_quiet();
  } else if (warp_id == 1) {
    for (uint i = lane_id; i < inter_node_rank_n; i += THREAD_N_PER_WARP){
      const uint32_t src_rank = ((server_id + 1)* local_rank_n + i) % rank_n;
      // + 1 is because we've added a signal to all ranks to avoid no
      // communication.
      const int64_t recv_sz = __ldg(&inter_recver_transfer_sz[src_rank]) + 1;
      nvshmem_uint64_wait_until(&sync_signal[src_rank], NVSHMEM_CMP_EQ, recv_sz);
      sync_signal[src_rank] = 0;
    }
  }
}

}; // namespace

int launch_alltoallv(
  uint32_t this_rank,
  uint32_t rank_n_per_node,
  uint32_t rank_n,
  struct fanout_nvshmem_buffer_t * buf,
  struct internode_transfer_params_t * inter_params,
  // NOTE: this is a partial intra-node send.
  // Because we don't do optimizations for intra-node send,
  // in this function, we have rank_n_per_node == 0.
  // However, we need to memcpy locally (from send_nvshmem to recv_nvshmem on local rank).
  // To achieve this, we launch a memcpy from the host because
  // it utilizes DMA and is faster.
  // Hence, we need the local memcpy offset and size on CPU.
  int64_t my_rank_send_offset,
  int64_t my_rank_recv_offset,
  int64_t my_rank_send_sz,
  cudaStream_t stream
) {
  void* inter_args[] = {
    &this_rank,
    &rank_n_per_node,
    &rank_n,
    &buf->send_buffer,
    &buf->recv_buffer,
    &buf->sync_signal,
    &inter_params->sender_send_disp,
    &inter_params->sender_transfer_sz,
    &inter_params->sender_recv_disp,
    &inter_params->recver_transfer_sz,
  };
  dim3 inter_grid(1, 1, 1), inter_block(THREAD_N_PER_2WARP, 1, 1);
  // The local communication
  CUDACHECK(cudaMemcpyAsync(
    buf->recv_buffer + my_rank_recv_offset,
    buf->send_buffer + my_rank_send_offset,
    my_rank_send_sz,
    cudaMemcpyDeviceToDevice,
    stream
  ));
  CUDACHECK(cudaLaunchKernel(
    (void *)&spreadout_alltoallv_internode_kernel,
      inter_grid,
      inter_block, 
      inter_args,
      0,
      stream
    ));
  // CUDACHECK(cudaStreamSynchronize(stream1));
  return NVSHMEMX_SUCCESS;
}
}; // namespace attn