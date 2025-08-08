from d2.planner.equal_flops import plan_relocation, batch_to_items
import rich

K = 1024

def test_planner_equal_flops():
    rich.print("⚪ Testing planner equal flops...")
    items = batch_to_items([
        [16 * K] * 1,
        [8 * K] * 2,
        [4 * K] * 4,
        [2 * K] * 8, 
    ])
    expected_items = [
        {'q': 16384, 'kv': 16384, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': True},
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 0, 'src_gpuid': 1, 'is_original': True},
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 1, 'src_gpuid': 1, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 1, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 2, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 3, 'src_gpuid': 2, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 1, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 2, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 3, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 4, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 5, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 6, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 7, 'src_gpuid': 3, 'is_original': True}
    ]
    for item in expected_items:
        assert item in items, f"item = {item} not in items: {expected_items = }\n{items = }"
    
    replanned_items = plan_relocation(items, verbose=False, plot=False)
    expected_replanned_items = [
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 0, 'src_gpuid': 1, 'is_original': True},
        {'q': 8192, 'kv': 8192, 'gpuid': 1, 'seqid': 1, 'src_gpuid': 1, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 1, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 2, 'src_gpuid': 2, 'is_original': True},
        {'q': 4096, 'kv': 4096, 'gpuid': 2, 'seqid': 3, 'src_gpuid': 2, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 1, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 2, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 3, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 4, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 5, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 6, 'src_gpuid': 3, 'is_original': True},
        {'q': 2048, 'kv': 2048, 'gpuid': 3, 'seqid': 7, 'src_gpuid': 3, 'is_original': True},
        {'q': 3754, 'kv': 3754, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 1706, 'kv': 5460, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 2730, 'kv': 8190, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 2734, 'kv': 10924, 'gpuid': 3, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 1706, 'kv': 12630, 'gpuid': 2, 'seqid': 0, 'src_gpuid': 0, 'is_original': False},
        {'q': 3754, 'kv': 16384, 'gpuid': 0, 'seqid': 0, 'src_gpuid': 0, 'is_original': False}
    ]
    for item in expected_replanned_items:
        assert item in replanned_items, f"item = {item} not in replanned_items: {expected_replanned_items = }\n{replanned_items = }"

    rich.print("✅ Testing planner equal flops passed")
    pass


if __name__ == "__main__":
    test_planner_equal_flops()