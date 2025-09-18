
for tolerance_factor in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    planner = Planner(world_size, parallel_config, model_config=model_config, tolerance_factor=tolerance_factor)
    
    verbose = (rank % 8 == 0)
    fa2a_metadata_0, as_attn_metadata_0, mlp_shard_len_0 = planner.plan(_items_0, is_resend_qkv=resend_qkv, verbose=verbose)
    fa2a_metadata_1, as_attn_metadata_1, mlp_shard_len_1 = planner.plan(_items_1, is_resend_qkv=resend_qkv, verbose=verbose)


    if verbose:
        qkv_fwd_fa2a_metadata__send_transfer_sz_mb = fa2a_metadata_0[0].fa2a_metadata[1] // 1024 // 1024
        qkv_fwd_fa2a_metadata__recv_transfer_sz_mb = fa2a_metadata_0[0].fa2a_metadata[3] // 1024 // 1024
        attn_out_fwd_fa2a_metadata__send_transfer_sz_mb = fa2a_metadata_0[1].fa2a_metadata[1] // 1024 // 1024
        attn_out_fwd_fa2a_metadata__recv_transfer_sz_mb = fa2a_metadata_0[1].fa2a_metadata[3] // 1024 // 1024
                
        # Print qkv_fwd_fa2a_metadata
        rich.print(f"🟡 [Rank {rank}] qkv_fwd_fa2a_metadata.send_transfer_sz_mb = ", qkv_fwd_fa2a_metadata__send_transfer_sz_mb)
        rich.print(f"🟡 [Rank {rank}] qkv_fwd_fa2a_metadata.recv_transfer_sz_mb = ", qkv_fwd_fa2a_metadata__recv_transfer_sz_mb)
        
        # Print attn_out_fwd_fa2a_metadata
        rich.print(f"🟡 [Rank {rank}] attn_out_fwd_fa2a_metadata.send_transfer_sz_mb = ", attn_out_fwd_fa2a_metadata__send_transfer_sz_mb)
        rich.print(f"🟡 [Rank {rank}] attn_out_fwd_fa2a_metadata.recv_transfer_sz_mb = ", attn_out_fwd_fa2a_metadata__recv_transfer_sz_mb)

    # Check size:
    buffer_size = FastDispatcherWrapper.instance[0].buffer_size
    def _check_overflow(fa2a_metadata):
        send_sz = [torch.sum(m.fa2a_metadata[1][as_rank]).item() for m in fa2a_metadata]
        # send_sz + sender_recv_offset = sender_recv_last_token
        send_last_offset = [(m.fa2a_metadata[1] + m.fa2a_metadata[2])[as_rank] for m in fa2a_metadata]
        recv_sz = [torch.sum(m.fa2a_metadata[3][as_rank]).item() for m in fa2a_metadata]
        max_send_sz = max(send_sz)
        max_recv_sz = max(recv_sz)
        
        if rank % 8 == 0:
            rich.print(f"🟡 [Rank {rank}] Overflow check: {max_send_sz / 1024**3:.2f} GB, {max_recv_sz / 1024**3:.2f} GB recv size, {max(torch.max(o).item() for o in send_last_offset) / 1024**3:.2f} GB send last offset, {buffer_size / 1024**3:.2f} GB buffer size")

        max_size_provisioned = max(
            max_send_sz, max_recv_sz, max(torch.max(o).item() for o in send_last_offset)
        )
        if not (buffer_size >= max_size_provisioned):
            return False, max_size_provisioned
        return True, max_size_provisioned
        
        # assert buffer_size >= max_send_sz and buffer_size >= max_recv_sz, f"{buffer_size / 1024**3} GB buffer, {
        #     [s / 1024**3 for s in send_sz]} GB send sizes, {
        #     [sz / 1024**3 for sz in recv_sz]} GB recv sizes"
        # assert max(torch.max(o).item() for o in send_last_offset) <= buffer_size, f"{buffer_size / 1024**3} GB buffer, {[o / 1024**3 for o in send_last_offset]} GB send last offsets"

    check_0, max_size_provisioned_0 = _check_overflow(fa2a_metadata_0)
    check_1, max_size_provisioned_1 = _check_overflow(fa2a_metadata_1)
    max_size_provisioned = max(max_size_provisioned_0, max_size_provisioned_1) / 1024**3
    required_buffer_size.append(max_size_provisioned)
    
    if not (check_0 and check_1):
        rich.print(f"⚠️ [Rank {rank}] Overflow check failed for fa2a_metadata_0 or fa2a_metadata_1 with tolerance_factor {tolerance_factor} and buffer_size {buffer_size / 1024**3} GB. Retry...")
    else:
        did_pass_overflow_check = True
        break
