import pandas as pd
import numpy as np
import glob
import os
from scipy.spatial import cKDTree
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. The Clean Wave CSV you created earlier
WEATHER_CSV = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\data\weather_csvs\CLEAN_WAVES_2024.csv"

# 2. Input & Output Folders
INPUT_TRACKS_DIR = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\data\processed\cargo_dense_region"
OUTPUT_DIR = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\data\enriched_tracks_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def enrich_all_tracks():
    print("=== 1. LOADING WEATHER DATA (One-Time Load) ===")
    print(f"Reading {os.path.basename(WEATHER_CSV)}...")
    
    # Read the clean wave data
    df_weather = pd.read_csv(WEATHER_CSV)
    df_weather['time'] = pd.to_datetime(df_weather['time'], utc=True)
    
    # Sort for faster merging logic
    df_weather = df_weather.sort_values('time')
    
    print(f"Loaded {len(df_weather)} rows of weather data.")

    # --- BUILD SPATIAL INDEX (KDTree) ---
    print("Building Spatial Index (KDTree)...")
    unique_geo = df_weather[['latitude', 'longitude']].drop_duplicates().reset_index(drop=True)
    tree = cKDTree(unique_geo[['latitude', 'longitude']].values)
    
    print("\n=== 2. STARTING BATCH PROCESSING ===")
    track_files = glob.glob(os.path.join(INPUT_TRACKS_DIR, "*.csv"))
    print(f"Found {len(track_files)} files to process.")
    
    success_count = 0
    fail_count = 0

    # Loop through ALL files with a progress bar
    for file_path in tqdm(track_files, desc="Enriching Tracks", unit="file"):
        try:
            filename = os.path.basename(file_path)
            
            # Read Track
            df_track = pd.read_csv(file_path)
            if df_track.empty: continue

            # Standardize Ship Time
            # Handling the ISO format with potential microseconds
            df_track['Estimated_Timestamp'] = pd.to_datetime(
                df_track['Estimated_Timestamp'], 
                format='ISO8601', 
                utc=True
            )
            
            # --- STEP A: SPATIAL MATCHING ---
            # Find nearest weather grid point (k=1 means closest neighbor)
            distances, indices = tree.query(df_track[['LAT', 'LON']].values, k=1)
            
            # Assign the "Official" Weather Coordinates to the Ship Track
            # We use these snapped coords to look up the data
            df_track['weather_lat'] = unique_geo.iloc[indices]['latitude'].values
            df_track['weather_lon'] = unique_geo.iloc[indices]['longitude'].values

            # --- STEP B: MERGE DATA ---
            # Round ship time to nearest 3 Hours (since wave data is 3-hourly)
            df_track['weather_time_snap'] = df_track['Estimated_Timestamp'].dt.round('3h')
            
            # Merge track with weather on [lat, lon, time]
            merged = pd.merge(
                df_track,
                df_weather,
                left_on=['weather_lat', 'weather_lon', 'weather_time_snap'],
                right_on=['latitude', 'longitude', 'time'],
                how='left'
            )
            
            # --- STEP C: CLEANUP ---
            # Rename for clarity
            cols_to_keep = {
                'VHM0': 'Wave_Height',
                'VMDR': 'Wave_Dir',
                'VTPK': 'Wave_Period'
            }
            merged = merged.rename(columns=cols_to_keep)
            
            # Define columns to keep: Original Track Cols + New Weather Cols
            # (We drop the temp columns like 'weather_lat' used for matching)
            base_cols = [c for c in df_track.columns if c not in ['weather_lat', 'weather_lon', 'weather_time_snap']]
            weather_cols = ['Wave_Height', 'Wave_Dir', 'Wave_Period']
            
            # --- THE FIX: ADD .copy() ---
            # This creates a new independent dataframe, preventing SettingWithCopyWarning
            final_df = merged[base_cols + weather_cols].copy()
            
            # Fill NaNs with 0.0 (Assume calm water if matching failed)
            final_df[weather_cols] = final_df[weather_cols].fillna(0.0)

            # Save
            final_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
            success_count += 1

        except Exception as e:
            # print(f"Error on {filename}: {e}") # Uncomment if you want to see errors
            fail_count += 1
            continue

    print("\n" + "="*40)
    print(f"✅ BATCH COMPLETE")
    print(f"Successfully Enriched: {success_count}")
    print(f"Failed/Empty:        {fail_count}")
    print(f"Output Folder:       {OUTPUT_DIR}")
    print("="*40)

if __name__ == "__main__":
    enrich_all_tracks()