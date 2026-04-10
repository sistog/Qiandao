import numpy as np
import matplotlib.pyplot as plt

# Generate data
M = 256  # Window length, high value for smoothness
n = np.arange(M)  # Sample index axis
window = np.hanning(M)  # Hanning window values

# Plot curve
plt.figure(figsize=(10, 6)) # Define canvas size

# plt.plot connects points, creating the curve effect
# Using a dark purple (#3b0f70) and slightly thicker line for emphasis
plt.plot(n, window, color="#b57e05", linewidth=2)

# Set titles and labels for scanning
plt.title(f"Continuous Hanning Window Curve (M={M})")
plt.xlabel("Sample index (n)")
plt.ylabel("Amplitude")

# Configure axis limits for clarity
plt.xlim(-10, M + 9)
plt.ylim(0, 1.05)

# Add dashed grid lines
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("../graph/hanning_curve.png", dpi=600) # Uncomment to save high-res image
plt.show()