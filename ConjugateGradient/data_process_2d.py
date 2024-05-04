import numpy as np
import matplotlib.pyplot as plt

data1 = np.loadtxt('phi_mpi_periodic_boundary.txt')
data1_normalized = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1))

fig, ax = plt.subplots(figsize=(6, 6))

cax = ax.matshow(data1_normalized, interpolation='nearest', cmap='hot')
fig.colorbar(cax, ax=ax)
ax.set_title('Normalized Heatmap for CG')

plt.savefig('heatmap.png')
plt.show()
