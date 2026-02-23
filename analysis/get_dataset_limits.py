import pandas as pd
import glob
import os
import sys

# --- CONFIGURATION ---
TARGET_FOLDER = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\src\data\data\processed\long_voyages"

def get_limits():
    files = glob.glob(os.path.join(TARGET_FOLDER, "*.csv"))
    if not files:
        print("No files found! Check the path.")
        return

    print(f"Calculating Limits for {len(files)} tracks...")

    # Initialize with inverse values
    min_lat, max_lat = 90.0, -90.0
    min_lon, max_lon = 180.0, -180.0
    min_time = pd.Timestamp.max.replace(tzinfo=pd.Timestamp.now().tz) if pd.Timestamp.max.tzinfo else pd.Timestamp.max
    max_time = pd.Timestamp.min.replace(tzinfo=pd.Timestamp.now().tz) if pd.Timestamp.min.tzinfo else pd.Timestamp.min
    
    # We need timezone aware default for comparison since data is UTC
    min_time = pd.Timestamp.max.tz_localize('UTC')
    max_time = pd.Timestamp.min.tz_localize('UTC')

    for i, file in enumerate(files):
        try:
            # We read only the necessary columns
            df = pd.read_csv(file, usecols=['LAT', 'LON', 'Estimated_Timestamp'])
            if df.empty: continue

            # 1. Update Spatial Bounds (Fast Vectorized Ops)
            file_min_lat = df['LAT'].min()
            file_max_lat = df['LAT'].max()
            file_min_lon = df['LON'].min()
            file_max_lon = df['LON'].max()

            if file_min_lat < min_lat: min_lat = file_min_lat
            if file_max_lat > max_lat: max_lat = file_max_lat
            if file_min_lon < min_lon: min_lon = file_min_lon
            if file_max_lon > max_lon: max_lon = file_max_lon

            # 2. Update Temporal Bounds (THE FIX)
            # utc=True forces standardized parsing and suppresses the UserWarning
            df['Estimated_Timestamp'] = pd.to_datetime(df['Estimated_Timestamp'], utc=True)
            
            file_start = df['Estimated_Timestamp'].min()
            file_end = df['Estimated_Timestamp'].max()

            if file_start < min_time: min_time = file_start
            if file_end > max_time: max_time = file_end

            if i % 2000 == 0: print(f"Scanned {i}/{len(files)}...")

        except Exception as e:
            # print(f"Skipping error in {file}: {e}") # Uncomment to debug specific files
            continue

    print("\n" + "="*40)
    print("   DATASET BOUNDARIES (RL SPECS)")
    print("="*40)
    print(f"Total Tracks: {len(files)}")
    print("-" * 20)
    print("SPATIAL BOUNDING BOX (For Weather/Grid):")
    print(f"  North (Max Lat): {max_lat:.4f}")
    print(f"  South (Min Lat): {min_lat:.4f}")
    print(f"  East  (Max Lon): {max_lon:.4f}")
    print(f"  West  (Min Lon): {min_lon:.4f}")
    print("-" * 20)
    print("TEMPORAL RANGE:")
    print(f"  Earliest Date:   {min_time}")
    print(f"  Latest Date:     {max_time}")
    
    duration = max_time - min_time
    print(f"  Total Duration:  {duration.days} days")
    print("="*40)
    
    # Suggest a buffer for the RL Env
    print("\n[Recommendation] RL Environment Bounds (+1 Degree Buffer):")
    print(f"N: {max_lat+1:.2f}, S: {min_lat-1:.2f}, E: {max_lon+1:.2f}, W: {min_lon-1:.2f}")

if __name__ == "__main__":
    get_limits()