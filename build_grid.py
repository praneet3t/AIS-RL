import glob
import pandas as pd
import numpy as np
import scipy.ndimage as ndimage
from tqdm import tqdm

# Configuration
CSV_FOLDER_PATH = "/workspace/AIS-RL/data/enriched_tracks_final/*.csv"
GRID_SIZE = 200

print("Scanning files to determine boundaries...")
file_list = glob.glob(CSV_FOLDER_PATH)
min_lat, max_lat, min_lon, max_lon = 90.0, -90.0, 180.0, -180.0

# Fast pass to get boundaries
for file in tqdm(file_list[:500]):
    df = pd.read_csv(file, usecols=['LAT', 'LON'])
    if not df.empty:
        min_lat = min(min_lat, df['LAT'].min())
        max_lat = max(max_lat, df['LAT'].max())
        min_lon = min(min_lon, df['LON'].min())
        max_lon = max(max_lon, df['LON'].max())

# Add a 0.1 degree buffer
min_lat, max_lat = min_lat - 0.1, max_lat + 0.1
min_lon, max_lon = min_lon - 0.1, max_lon + 0.1

print(f"Bounds: Lat({min_lat:.2f}, {max_lat:.2f}), Lon({min_lon:.2f}, {max_lon:.2f})")

# Setup grid bins
lat_bins = np.linspace(min_lat, max_lat, GRID_SIZE + 1)
lon_bins = np.linspace(min_lon, max_lon, GRID_SIZE + 1)

traffic_counts = np.zeros((GRID_SIZE, GRID_SIZE))
wave_h_sum = np.zeros((GRID_SIZE, GRID_SIZE))
wave_p_sum = np.zeros((GRID_SIZE, GRID_SIZE))

print("Mapping AIS data to the 200x200 grid...")
for file in tqdm(file_list):
    df = pd.read_csv(file, usecols=['LAT', 'LON', 'Wave_Height', 'Wave_Period']).dropna()
    if df.empty: continue
    
    lat_idx = np.clip(np.digitize(df['LAT'].values, lat_bins) - 1, 0, GRID_SIZE - 1)
    lon_idx = np.clip(np.digitize(df['LON'].values, lon_bins) - 1, 0, GRID_SIZE - 1)
    
    for i in range(len(df)):
        x, y = lat_idx[i], lon_idx[i]
        traffic_counts[x, y] += 1
        wave_h_sum[x, y] += df['Wave_Height'].values[i]
        wave_p_sum[x, y] += df['Wave_Period'].values[i]

# Process Features
safe_counts = np.where(traffic_counts == 0, 1, traffic_counts)
grid_wave_h = wave_h_sum / safe_counts
grid_wave_p = wave_p_sum / safe_counts

# Fill empty sea cells with mean wave data so math doesn't break
grid_wave_h[traffic_counts == 0] = np.mean(grid_wave_h[traffic_counts > 0])
grid_wave_p[traffic_counts == 0] = np.mean(grid_wave_p[traffic_counts > 0])

# Smooth traffic to create navigable lanes (Gaussian KDE)
grid_traffic = ndimage.gaussian_filter(traffic_counts, sigma=1.5)

# Z-score Normalization
def normalize(grid):
    return (grid - np.mean(grid)) / (np.std(grid) + 1e-8)

phi_s = np.stack([normalize(grid_wave_h), normalize(grid_wave_p), normalize(grid_traffic)], axis=-1)

# Calculate Expert Expectations (Human Average)
expert_expectations = np.zeros(3)
for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        if traffic_counts[x, y] > 0:
            expert_expectations += phi_s[x, y] * traffic_counts[x, y]
expert_expectations /= np.sum(traffic_counts)

# Save the grid state and bounds
np.savez("irl_grid_data.npz", 
         phi_s=phi_s, 
         phi_E=expert_expectations, 
         bounds=np.array([min_lat, max_lat, min_lon, max_lon]))

print("Done. Saved 'irl_grid_data.npz'.")