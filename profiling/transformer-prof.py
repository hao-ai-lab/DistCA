import modal

image = modal.Image.debian_slim().pip_install(
    "torch",
    "transformers",
)

app = modal.App(name="transformer-prof")


K = 1024
M = 1024 ** 2

@app.function(image=image, gpu="H100:1", timeout=60)
def profile_qwen3(
    ctx_len: int = K * 4,
    model_id: str = "Qwen/Qwen3-235B-A22B"
):
    import time
    import torch
    import transformers.modeling_utils
    from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

    # 94 layers originally  [oai_citation:0‡Hugging Face](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json?utm_source=chatgpt.com)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg.num_hidden_layers = 1

    torch.set_default_device("cuda")
    torch.manual_seed(42)              # reproducible randomness
    with torch.no_grad():
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    profile = {}  # name → dict(start, end, params)

    def want_hook(name, module):
        # leaf = no children; skip attention classes (FlashAttention2, QwenAttention, …)
        is_leaf = len(list(module.children())) == 0
        is_attn = "attn" in module.__class__.__name__.lower() or \
                "attention" in module.__class__.__name__.lower()
        return is_leaf 
        # return is_leaf and not is_attn

    for name, m in model.named_modules():
        if not want_hook(name, m):
            continue

        # forward-pre & forward hooks share the closure variable 'name'
        def pre_hook(mod, inp, name=name):
            profile[name] = {"start": time.perf_counter(),
                            "params": sum(p.numel() for p in mod.parameters())}

        def post_hook(mod, inp, out, name=name):
            torch.cuda.synchronize()
            profile[name]["end"] = time.perf_counter()

        m.register_forward_pre_hook(pre_hook, prepend=True)
        m.register_forward_hook(post_hook)

    old_attn_funcs = transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping
    
    # Hook the attention functions too
    new_attn_funcs = {}
    for k, v in old_attn_funcs.items():
        def new_func(*args, **kwargs):
            start = time.perf_counter()
            ret = v(*args, **kwargs)
            torch.cuda.synchronize()
            end = time.perf_counter()
            profile["model.layers.0.self_attn.attn"] = {
                "start": start,
                "params": 0,
                "end": end,
            }
            return ret
        new_attn_funcs[k] = new_func

    transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping = new_attn_funcs

    
    # Introduce the input
    inp = tok(" a" * ctx_len, return_tensors="pt").to(model.device)
    ctx_len = len(inp.input_ids[0])

    with torch.no_grad():
        model(**inp)                        # warm-up (CUDA kernels/JIT)
        torch.cuda.synchronize()

        start = time.perf_counter()
        model(**inp)
        torch.cuda.synchronize()
        total = time.perf_counter() - start
        total *= 1e3


    # Profile the model
    rows = []
    for name, rec in profile.items():
        dur_ms = (rec["end"] - rec["start"]) * 1e3
        rows.append((dur_ms, name, rec["params"]))

    print(f"{'module':45}  latency  params")
    print("-"*70)

    total_layer0_time = 0
    expert_time = 0
    expert_params = 0

    layer_attn_time = 0
    for dur, name, nparam in rows:
        # if "layers.0" not in name:
        #     continue
        if "experts" in name:
            expert_time += dur
            expert_params += nparam
        else:
            print(f"{name:45}  {dur:7.3f} ms  {nparam/1e6:7.2f} M")
        
        if name == "model.layers.0.self_attn.attn":
            layer_attn_time += dur

        if "layers.0" not in name:
            continue

        total_layer0_time += dur

    layer_mlp_time = total_layer0_time - layer_attn_time

    full_result = dict(
        model_id=model_id,
        ctx_len=ctx_len,
        layer_attn_time=layer_attn_time,
        layer_mlp_time=layer_mlp_time, 
        total_layer0_time=total_layer0_time,
        expert_time=expert_time,
        expert_params=expert_params,
        full_forward_time=total,
        details=rows,
    )

    mlp_only_result = dict(
        model_id=model_id,
        ctx_len=ctx_len,
        layer_mlp_time=layer_mlp_time, 
    )

    return full_result, mlp_only_result


@app.local_entrypoint()
def main(
    min_ctx_len: int = 64,
    max_ctx_len: int = M * 1,
    step_factor: int = 2,
    model_id: str = "Qwen/Qwen3-235B-A22B",
):
    import json

    filename = f"compute-profile.jsonl"
    mlp_only_filename = f"compute-profile-mlp-only.csv"

    with open(mlp_only_filename, "w") as f:
        f.write("model_id,ctx_len,layer_mlp_time\n")

    with open(filename, "w") as f, open(mlp_only_filename, "w") as f_mlp:
        ctx_len = min_ctx_len
        while ctx_len <= max_ctx_len:
            full_result, mlp_only_result = profile_qwen3.remote(
                ctx_len=ctx_len,
                model_id=model_id,
            )
            json.dump(full_result, f)
            f.write("\n")
            f.flush()

            print(mlp_only_result)
            f_mlp.write(f"{model_id},{ctx_len},{mlp_only_result['layer_mlp_time']:.2f}\n")
            f_mlp.flush()
            
            ctx_len *= step_factor



if __name__ == "__main__":
    main()