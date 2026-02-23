import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load your math
reward_matrix = np.load("final_reward_grid.npy")

# Convert Rewards to "Costs" (Invert it so high reward = deep valley)
cost_matrix = np.max(reward_matrix) - reward_matrix

# Create X and Y coordinates for the 3D grid
x = np.linspace(0, cost_matrix.shape[1], cost_matrix.shape[1])
y = np.linspace(0, cost_matrix.shape[0], cost_matrix.shape[0])
X, Y = np.meshgrid(x, y)

# Set up the 3D plot
fig = plt.figure(figsize=(14, 10), facecolor='black')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Hide the ugly grid lines and axis numbers
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
ax.axis('off')

# Plot the 3D surface
surf = ax.plot_surface(X, Y, cost_matrix, cmap='magma', linewidth=0, antialiased=True, alpha=0.9)

# Adjust the viewing angle to make it look dramatic
ax.view_init(elev=45, azim=-120)

plt.title("The Human Prior: 3D Navigational Cost Surface", color='white', fontsize=18, pad=20)
plt.savefig('3d_cost_surface.png', dpi=300, bbox_inches='tight', facecolor='black')
print("Saved mind-blowing 3D map as '3d_cost_surface.png'")