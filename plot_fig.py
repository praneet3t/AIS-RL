import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import scipy.ndimage as ndimage

# ==========================================
# 1. LOAD DATA & SET ACADEMIC STYLE
# ==========================================
print("📥 Loading data for high-res plotting...")
data = np.load("irl_grid_data.npz")
phi_s = data['phi_s']  # (100, 100, 3) -> [Wave_H, Wave_P, Traffic]

# Reconstruct raw grids from normalized features
wave_h = phi_s[:, :, 0]
traffic = phi_s[:, :, 2]

# Load or Mock the reward grid
try:
    reward = np.load("final_reward_grid.npy")
    print("✅ Found final_reward_grid.npy!")
except:
    print("⚠️ final_reward_grid.npy not found. Run Step 3 first! Mocking data...")
    reward = traffic - wave_h 

# Set sleek academic fonts and parameters
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300 
})

# ==========================================
# 2. CREATE THE 3-PANEL FIGURE
# ==========================================
fig = plt.figure(figsize=(18, 6))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.15)

# Ensure traffic is positive for log scaling
traffic_plot = traffic - np.min(traffic) + 1e-3

# --- PANEL 1: Traffic Density (KDE) ---
ax1 = plt.subplot(gs[0])
im1 = ax1.imshow(traffic_plot.T, cmap='magma', origin='lower', norm=LogNorm(vmin=1e-2, vmax=np.max(traffic_plot)))
ax1.set_title("A) Expert Trajectory Density (KDE)", loc='left', fontweight='bold')
ax1.set_xticks([]); ax1.set_yticks([])
cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04) # Fixed: ax=ax1
cbar1.set_label('Log Density', rotation=270, labelpad=15)

# --- PANEL 2: Environment + Behavior Overlay ---
ax2 = plt.subplot(gs[1])
im2 = ax2.imshow(wave_h.T, cmap='Blues', origin='lower')
levels = np.percentile(traffic_plot, [75, 90, 95, 99])
ax2.contour(traffic_plot.T, levels=levels, colors='red', linewidths=[0.5, 1.0, 1.5, 2.0], alpha=0.8)
ax2.set_title("B) Env Hazards & Routing Contours", loc='left', fontweight='bold')
ax2.set_xticks([]); ax2.set_yticks([])
cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04) # Fixed: ax=ax2
cbar2.set_label('Normalized Wave Height ($H_s$)', rotation=270, labelpad=15)

# --- PANEL 3: The Learned Reward (IRL Output) ---
ax3 = plt.subplot(gs[2])
im3 = ax3.imshow(reward.T, cmap='viridis', origin='lower')
ax3.set_title("C) Recovered Navigation Prior $R(s)$", loc='left', fontweight='bold')
ax3.set_xticks([]); ax3.set_yticks([])
cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04) # Fixed: ax=ax3
cbar3.set_label('Learned Utility (Reward)', rotation=270, labelpad=15)

# ==========================================
# 3. SAVE HIGH-RES OUTPUT
# ==========================================
plt.savefig("nature_fig_irl_pipeline.png", bbox_inches='tight', dpi=300)
print("✅ Saved publication-ready figure: nature_fig_irl_pipeline.png")