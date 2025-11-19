"""Test that FFT plots are square when real-space ranges are equal"""

from stm_toolkit import SXMFile
import matplotlib.pyplot as plt

# Test with RS52_0073.sxm which has equal ranges but different pixel densities
sxm = SXMFile(r'C:\Users\hnhua\Documents\Code\tests\RS52_0073.sxm')
sxm.load()

print(f"Real space: {sxm.x_range} nm x {sxm.y_range} nm")
print(f"Pixels: {sxm.x_pixels} x {sxm.y_pixels}")
print(f"Pixel sizes: {sxm.x_range/sxm.x_pixels:.4f} nm/pix x {sxm.y_range/sxm.y_pixels:.4f} nm/pix")

# Create plotter and FFT
plotter = sxm.plot(channel='A (both)', direction=0)
plotter.fft()

# Get the extent
xlim = plotter.fft_ax.get_xlim()
ylim = plotter.fft_ax.get_ylim()

print(f"\nFFT extent:")
print(f"  X: [{xlim[0]:.4f}, {xlim[1]:.4f}] nm^-1")
print(f"  Y: [{ylim[0]:.4f}, {ylim[1]:.4f}] nm^-1")

# Check if square
x_range = abs(xlim[1] - xlim[0])
y_range = abs(ylim[1] - ylim[0])
print(f"\nFFT ranges: {x_range:.4f} nm^-1 x {y_range:.4f} nm^-1")

if abs(x_range - y_range) < 0.01:
    print("[SUCCESS] FFT plot is square!")
else:
    print(f"[WARNING] FFT plot is not square (difference: {abs(x_range - y_range):.4f})")

# Save
plotter.fft_fig.savefig('test_fft_square.png', dpi=150)
print("\nSaved test_fft_square.png")

plotter.close()
plt.close(plotter.fft_fig)

