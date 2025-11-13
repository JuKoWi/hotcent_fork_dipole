import numpy as np
import matplotlib.pyplot as plt

mse = np.load("mse_grid_scan_quick.npy")
mae = np.load("mae_grid_scan_quick.npy")

plt.imshow(mae, cmap='viridis', origin='lower')
plt.show()
