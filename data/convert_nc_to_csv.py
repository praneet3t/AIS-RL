import xarray as xr
import pandas as pd
import os
import glob

# --- CONFIGURATION ---
# The folder where your .nc files are
INPUT_DIR = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\data\weather"
OUTPUT_DIR = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\data\weather_csvs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_waves_fresh():
    print("=== STARTING FRESH CONVERSION ===")
    
    # 1. Find the Wave NetCDF File
    wave_files = glob.glob(os.path.join(INPUT_DIR, "*wav*.nc"))
    if not wave_files:
        raise FileNotFoundError("❌ No Wave .nc file found in the weather folder!")
    
    file_path = wave_files[0]
    print(f"Target File: {os.path.basename(file_path)}")

    try:
        # 2. Open the Dataset
        ds = xr.open_dataset(file_path)
        
        # 3. Select only the critical variables
        # VHM0 = Wave Height, VMDR = Direction, VTPK = Period
        # We drop others to keep the CSV smaller and cleaner
        ds_subset = ds[['VHM0', 'VMDR', 'VTPK']]
        
        print("Flattening to DataFrame... (This uses RAM)")
        # 4. Convert to DataFrame (Lat/Lon/Time become columns)
        df = ds_subset.to_dataframe().reset_index()
        
        total_rows = len(df)
        print(f"Total Grid Points: {total_rows}")

        # 5. REMOVE LAND (NaNs)
        # If Wave Height (VHM0) is NaN, it's land. We drop it.
        print("Filtering out Land (NaN values)...")
        df_clean = df.dropna(subset=['VHM0'])
        
        valid_rows = len(df_clean)
        print(f"Valid Water Points: {valid_rows}")
        
        if valid_rows == 0:
            print("❌ CRITICAL ERROR: The file contains no valid data (All NaNs).")
            print("   Check your Bounding Box coordinates.")
            return

        # 6. Format Time
        # Ensure it is standard ISO format for easy reading later
        df_clean['time'] = pd.to_datetime(df_clean['time'])

        # 7. Save
        output_filename = "CLEAN_WAVES_2024.csv"
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"Saving to {save_path}...")
        df_clean.to_csv(save_path, index=False)
        
        # 8. VERIFICATION PRINT
        print("\n=== VERIFICATION (First 3 Rows) ===")
        print(df_clean[['time', 'latitude', 'longitude', 'VHM0']].head(3))
        print("===================================")
        print("✅ Conversion Successful.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    convert_waves_fresh()