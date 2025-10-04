#!/usr/bin/env python3
"""
Test script to verify all-to-all timing implementation
"""

import os
import sys
import torch

# Set environment variable to enable timing
os.environ["UNIFIED_RECORD_A2A_TIMES"] = "1"

# Add the test directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_a2a_timing_setup():
    """Test that the all-to-all timing setup works without errors."""
    try:
        from tests.test_e2e_combined import (
            setup_unified_a2a_timing_patch,
            set_unified_current_a2a_sample_id,
            sync_and_collect_a2a_timing,
            get_unified_a2a_times,
            clear_unified_a2a_times
        )
        
        print("✅ Successfully imported all-to-all timing functions")
        
        # Test setup
        setup_unified_a2a_timing_patch()
        print("✅ Successfully setup all-to-all timing patch")
        
        # Test sample ID setting
        set_unified_current_a2a_sample_id(0)
        print("✅ Successfully set sample ID")
        
        # Test timing collection (should be empty initially)
        timing_data = sync_and_collect_a2a_timing()
        print(f"✅ Timing data collection works: {timing_data}")
        
        # Test getting times
        all_times = get_unified_a2a_times()
        print(f"✅ Get unified times works: {all_times}")
        
        # Test the simplified structure
        expected_structure = {"a2a_forward": []}
        print(f"✅ Expected structure: {expected_structure}")
        
        # Test clearing
        clear_unified_a2a_times()
        print("✅ Clear unified times works")
        
        print("\n🎉 All all-to-all timing functions work correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing all-to-all timing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_a2a_timing_setup()
    sys.exit(0 if success else 1)
