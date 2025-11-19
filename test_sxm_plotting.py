"""
Test script for plotting SXM files.
"""

import sys
from pathlib import Path
from stm_toolkit import SXMFile
import matplotlib.pyplot as plt

# Test directory
test_dir = Path(r"C:\Users\hnhua\Documents\Code\tests")

# Test with one file
test_file = test_dir / "RS52_0073.sxm"
print(f"Testing plotting with: {test_file.name}")

try:
    # Load file
    sxm = SXMFile(test_file)
    sxm.load()
    
    print(f"Channels available: {sxm.get_channel_names()}")
    
    # Test plotting first channel
    channel = sxm.get_channel_names()[1]
    print(f"\nPlotting channel: {channel}")
    
    # Create plotter
    plotter = sxm.plot(channel=channel, direction=0, subtract_plane=False)
    plotter.set_title(f"{test_file.name} - {channel}")
    
    # Save plot
    output_file = "test_plot.png"
    plotter.save(output_file, dpi=150)
    print(f"Saved plot to: {output_file}")
    
    # Test FFT
    print("Creating FFT plot...")
    plotter.fft()
    fft_output = "test_fft.png"
    plotter.fft_fig.savefig(fft_output, dpi=150)
    print(f"Saved FFT plot to: {fft_output}")
    
    # Close plots
    plotter.close()
    plt.close(plotter.fft_fig)
    
    print("\n[SUCCESS] Plotting test completed!")
    
except Exception as e:
    print(f"[ERROR] Error during plotting: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

