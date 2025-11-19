"""
Test script for loading SXM files from the tests directory.
"""

import sys
from pathlib import Path
from stm_toolkit import SXMFile

# Test directory
test_dir = Path(r"C:\Users\hnhua\Documents\Code\tests")

# Find all .sxm files
sxm_files = list(test_dir.glob("*.sxm"))
print(f"Found {len(sxm_files)} .sxm files in test directory\n")

# Test loading each file
for sxm_path in sxm_files:
    print(f"=" * 80)
    print(f"Testing: {sxm_path.name}")
    print("=" * 80)
    
    try:
        # Create SXM file object
        sxm = SXMFile(sxm_path)
        
        # Load the file
        print("Loading file...")
        raw_data = sxm.load()
        
        # Print header information
        print("\nHeader Information:")
        print(f"  X pixels: {sxm.x_pixels}")
        print(f"  Y pixels: {sxm.y_pixels}")
        print(f"  X range: {sxm.x_range:.2f} nm")
        print(f"  Y range: {sxm.y_range:.2f} nm")
        print(f"  Gate voltage: {sxm.get_gate_voltage():.3f} V")
        print(f"  Bias: {sxm.get_bias():.3f} V")
        
        # Print channel information
        print(f"\nChannels ({len(sxm.get_channel_names())}):")
        for i, channel in enumerate(sxm.get_channel_names(), 1):
            print(f"  {i}. {channel}")
            if channel in sxm.data:
                forward = sxm.data[channel][0]
                backward = sxm.data[channel][1]
                print(f"     Forward shape: {forward.shape}, range: [{forward.min():.3e}, {forward.max():.3e}]")
                print(f"     Backward shape: {backward.shape}, range: [{backward.min():.3e}, {backward.max():.3e}]")
        
        # Test processing
        print("\nTesting processing...")
        processed = sxm.process(background_subtract=False, fft=False)
        print("  [OK] Basic processing successful")
        
        # Test background subtraction
        processed = sxm.process(background_subtract=True, background_method='plane', fft=False)
        print("  [OK] Plane subtraction successful")
        
        processed = sxm.process(background_subtract=True, background_method='line', fft=False)
        print("  [OK] Line-by-line subtraction successful")
        
        # Test FFT
        processed = sxm.process(background_subtract=False, fft=True)
        print("  [OK] FFT computation successful")
        
        print(f"\n[SUCCESS] Successfully loaded and processed {sxm_path.name}\n")
        
    except Exception as e:
        print(f"[ERROR] Error loading {sxm_path.name}: {type(e).__name__}: {e}\n")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("Test completed!")

