#!/usr/bin/env python3
"""
Example usage of timemodule.py allreduce and allgather functions
"""

from timemodule import get_allreduce_time, get_allgather_time

def example_usage():
    print("Example Usage of AllReduce and AllGather Time Functions")
    print("=" * 60)
    
    # Example: Getting communication times for different scenarios
    scenarios = [
        ("Small model weights", 2, 1024),
        ("Medium model weights", 4, 1048576),  # 1M elements
        ("Large model weights", 8, 134217728),  # 128M elements
        ("Gradient sync", 2, 2097152),  # 2M elements
        ("Parameter server", 4, 67108864),  # 64M elements
    ]
    
    print("\nCommunication Times for Different Scenarios:")
    print("-" * 60)
    
    for description, world_size, nelem in scenarios:
        allreduce_time = get_allreduce_time(world_size, nelem)
        allgather_time = get_allgather_time(world_size, nelem)
        
        print(f"\n{description}:")
        print(f"  World size: {world_size} GPUs")
        print(f"  Elements: {format_nelem(nelem)}")
        print(f"  AllReduce time: {allreduce_time:.3f} ms")
        print(f"  AllGather time: {allgather_time:.3f} ms")
        print(f"  Ratio (AG/AR): {allgather_time/allreduce_time:.2f}x")
    
    # Example: Comparing scaling with world size
    print("\n" + "=" * 60)
    print("Scaling Analysis: How does communication time change with world size?")
    print("-" * 60)
    
    nelem = 1048576  # 1M elements
    print(f"\nFixed problem size: {format_nelem(nelem)} elements")
    print("World Size | AllReduce (ms) | AllGather (ms) | AR Speedup | AG Speedup")
    print("-" * 70)
    
    baseline_ar = get_allreduce_time(2, nelem)
    baseline_ag = get_allgather_time(2, nelem)
    
    for world_size in [2, 4, 8]:
        ar_time = get_allreduce_time(world_size, nelem)
        ag_time = get_allgather_time(world_size, nelem)
        ar_speedup = baseline_ar / ar_time
        ag_speedup = baseline_ag / ag_time
        
        print(f"    {world_size:2d}     |     {ar_time:6.3f}     |     {ag_time:6.3f}     |   {ar_speedup:5.2f}x   |   {ag_speedup:5.2f}x")
    
    # Example: Comparing scaling with problem size
    print("\n" + "=" * 60)
    print("Problem Size Scaling: How does communication time change with data size?")
    print("-" * 60)
    
    world_size = 4  # 4 GPUs
    print(f"\nFixed world size: {world_size} GPUs")
    print("Problem Size | AllReduce (ms) | AllGather (ms) | AR Throughput | AG Throughput")
    print("-" * 80)
    
    base_nelem = 1024
    for multiplier in [1, 4, 16, 64, 256, 1024]:
        nelem = base_nelem * multiplier
        try:
            ar_time = get_allreduce_time(world_size, nelem)
            ag_time = get_allgather_time(world_size, nelem)
            ar_throughput = nelem / ar_time / 1000  # K elements per ms
            ag_throughput = nelem / ag_time / 1000  # K elements per ms
            
            print(f" {format_nelem(nelem):>10s}  |     {ar_time:6.3f}     |     {ag_time:6.3f}     |    {ar_throughput:6.1f}K/ms |    {ag_throughput:6.1f}K/ms")
        except Exception as e:
            print(f" {format_nelem(nelem):>10s}  | Error: {e}")

if __name__ == "__main__":
    # Add the format_nelem function locally since it's not in timemodule
    def format_nelem(nelem):
        """Format number of elements in human readable format"""
        K = 1024
        M = K ** 2
        G = K ** 3
        
        if nelem >= G:
            return f"{nelem/G:.2f}G"
        elif nelem >= M:
            return f"{nelem/M:.2f}M"
        elif nelem >= K:
            return f"{nelem/K:.2f}K"
        else:
            return f"{nelem:.0f}"
    
    # Patch it into the module for this example
    import timemodule
    timemodule.format_nelem = format_nelem
    
    example_usage() 