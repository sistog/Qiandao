import matplotlib.pyplot as plt
import numpy as np

f_raw = np.linspace(0, 1, 8000)
f_mel = 2595 * np.log10(1 + f_raw / 700)
plt.figure(figsize=(10, 4))
plt.plot(f_raw, f_mel, color="#336fe7", linewidth=2)
plt.title("Mel Scale")
plt.xlabel("频率 (Hz)")
plt.ylabel("Mel 频率")
plt.xlim(0, 8000)
plt.ylim(0, 4000)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("mel_scale.png", dpi=600)
plt.show()
