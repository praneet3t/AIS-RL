import pandas as pd
import glob
import os
import shutil
import numpy as np

# --- CONFIGURATION ---
INPUT_FOLDER = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\src\data\data\processed\cargo_only"
DENSE_OUTPUT_FOLDER = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\data\processed\cargo_dense_region"

os.makedirs(DENSE_OUTPUT_FOLDER, exist_ok=True)

def find_and_filter_density():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Scanning {len(files)} tracks to find the densest region...")

    # We use a simple 2D histogram to find the hotspot
    all_lats = []
    all_lons = []
    
    # Sample every 5th file to speed up the "search" phase
    for i in range(0, len(files), 5):
        df = pd.read_csv(files[i], usecols=['LAT', 'LON'])
        all_lats.extend(df['LAT'].tolist())
        all_lons.extend(df['LON'].tolist())

    # Calculate 2D Histogram (1 degree resolution)
    heatmap, xedges, yedges = np.histogram2d(all_lons, all_lats, bins=[360, 180], range=[[-180, 180], [-90, 90]])
    
    # Find the peak coordinate
    idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    peak_lon = xedges[idx[0]]
    peak_lat = yedges[idx[1]]

    # Define a manageable 10x10 degree Bounding Box around the peak
    # You can change 'offset' to 2.5 for a 5x5 degree box if you want it even smaller
    offset = 5.0 
    bbox = {
        'min_lat': peak_lat - offset,
        'max_lat': peak_lat + offset,
        'min_lon': peak_lon - offset,
        'max_lon': peak_lon + offset
    }

    print(f"\nFound Peak Hotspot near: {peak_lat:.2f}N, {peak_lon:.2f}W")
    print(f"Target Bounding Box: {bbox}")

    # Now, move files that touch this box
    count = 0
    for file in files:
        df = pd.read_csv(file, usecols=['LAT', 'LON'])
        # Check if any part of the track enters the box
        inside = df[
            (df['LAT'] >= bbox['min_lat']) & (df['LAT'] <= bbox['max_lat']) &
            (df['LON'] >= bbox['min_lon']) & (df['LON'] <= bbox['max_lon'])
        ]
        
        if not inside.empty:
            shutil.copy(file, os.path.join(DENSE_OUTPUT_FOLDER, os.path.basename(file)))
            count += 1
            if count % 100 == 0: print(f"Isolated {count} regional tracks...")

    print(f"\n--- SUCCESS ---")
    print(f"Isolated {count} tracks in the densest sub-region.")
    print(f"New Folder: {DENSE_OUTPUT_FOLDER}")
    return bbox

if __name__ == "__main__":
    bbox_results = find_and_filter_density()