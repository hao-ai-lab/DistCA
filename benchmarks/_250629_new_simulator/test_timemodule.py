import timemodule as tm

def test_timing_functions():
    # Test configurations
    test_configs = [
        (1, 1, 32768),  # Exact match from data
        (2, 2, 32768),  # Exact match from data
        (1, 1, 40000),  # Should interpolate between 32768 and 65536
        (4, 4, 50000),  # Should interpolate
        (8, 8, 32768),  # Exact match from data
        (1, 2, 45000),  # Should interpolate
    ]
    
    print("Testing timing functions with various configurations:")
    print("\nFormat: (tp, cp, seq_len) -> time in ms")
    
    print("\n=== Attention Time Tests ===")
    for tp, cp, seq_len in test_configs:
        try:
            time = tm.get_attn_time(tp, cp, seq_len)
            print(f"({tp}, {cp}, {seq_len}) -> {time:.4f}ms")
        except Exception as e:
            print(f"({tp}, {cp}, {seq_len}) -> Error: {str(e)}")
    
    print("\n=== MLP Time Tests ===")
    for tp, cp, seq_len in test_configs:
        try:
            time = tm.get_mlp_time(tp, cp, seq_len)
            print(f"({tp}, {cp}, {seq_len}) -> {time:.4f}ms")
        except Exception as e:
            print(f"({tp}, {cp}, {seq_len}) -> Error: {str(e)}")
    
    print("\n=== AllGather Time Tests ===")
    for tp, cp, seq_len in test_configs:
        try:
            time = tm.get_allgather_time(tp, cp, seq_len)
            print(f"({tp}, {cp}, {seq_len}) -> {time:.4f}ms")
        except Exception as e:
            print(f"({tp}, {cp}, {seq_len}) -> Error: {str(e)}")
    
    # Test interpolation consistency
    print("\n=== Interpolation Consistency Check ===")
    print("Testing if interpolated values are between their bounds")
    
    tp, cp = 1, 1
    x1, x2 = 32768, 65536  # Known sequence lengths from data
    x_interp = 45000  # Point to interpolate
    
    try:
        y1 = tm.get_attn_time(tp, cp, x1)
        y2 = tm.get_attn_time(tp, cp, x2)
        y_interp = tm.get_attn_time(tp, cp, x_interp)
        
        print(f"\nAttention time interpolation check:")
        print(f"Lower bound  ({x1}): {y1:.4f}ms")
        print(f"Interpolated ({x_interp}): {y_interp:.4f}ms")
        print(f"Upper bound  ({x2}): {y2:.4f}ms")
        
        if y1 <= y_interp <= y2:
            print("✓ Interpolated value is within bounds")
        else:
            print("✗ Interpolated value is outside bounds!")
            
    except Exception as e:
        print(f"Error during interpolation check: {str(e)}")

if __name__ == "__main__":
    test_timing_functions() 