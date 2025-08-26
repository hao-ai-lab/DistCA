# %%
import simpy
# %%
env = simpy.Environment()
# %%
num_stages = 4
K = 1024
inboxes = [simpy.PriorityStore(env) for _ in range(num_stages)]
# %%
batches = [
    [64 * K],
    [32 * K] * 2,
    [16 * K] * 4,
    [8 * K] * 8,
]
# %%

def stage_with_signal(
    env, idx, inbox, 
    next_inbox, prev_inbox, 
    num_microbatches, 
    done_counter, log_data, 
    nlayers=1,
):
    while done_counter[idx] < num_microbatches:
        _, kind, m, pingpong_batches = yield inbox.get()
        ping_batch, pong_batch = pingpong_batches

        is_forward = (kind != "grad")
        is_first_stage = (idx == 0)
        is_last_stage = (idx == num_stages - 1)
        if is_forward and not is_last_stage:

            for nlayers in range(nlayers):
                t0 = env.now
                if is_first_stage:
                    # 2. pre-self-attention forward microbatch 0.
                    # _forward_pre_core_attn
                    # - compute: get qkv
                    _forward_pre_core_attn_0 = _forward_pre_core_attn(ping_batch)
                    # mlp_to_attn layout change (just mem copy) 
                    # tick_sync
                    
                    # 3. pre-attention forward of microbatch 1, mlp2attn all2all of microbatch 0
                    # all_to_all_0
                    # - comm: send kv 
                    all_ping_batch = _pre_mlp_to_attn_0 = _pre_mlp_to_attn(ping_batch, idx) 
                    # _forward_pre_core_attn_1 = ...
                    # _pre_mlp_to_attn.0
                    # tick_sync
                    
                    # 4. self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
                    # all_to_all_1
                    # _post_mlp_to_attn_0
                    # core_attn_0
                    # _pre_attn_to_mlp.0
                    # tick_sync

                    # 5. post-self-attention forward of microbatch 0, mlp2attn all2all of microbatch 1
                    # all_to_all_1
                    # _post_mlp_to_attn_1
                    # core_attn_1
                    # _pre_attn_to_mlp.1

                    # 6. mlp forward of microbatch 0, mlp2attn all2all of microbatch 1
                    # all_to_all_1
                    # _post_attn_to_mlp.0
                    # _forward_post_core_attn.0
                    # _pre_mlp_to_attn.1

                    pass
                pass
            
            
            
            pass
        elif is_forward and is_last_stage:
            # is forward pass, last stage
            pass
        elif not is_forward:
            # is backward pass, not last stage
            pass
        
        pass
    pass



# %%