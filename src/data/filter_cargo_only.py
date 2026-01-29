import pandas as pd
import os
import glob
import shutil

# --- CONFIGURATION ---
# The path you provided
INPUT_FOLDER = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\src\data\data\processed\long_voyages"
OUTPUT_FOLDER = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\src\data\data\processed\cargo_only"

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Cargo codes fallback (if VesselGroup is missing)
CARGO_CODES = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]

def filter_cargo():
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    print(f"Scanning {len(files)} files for Cargo vessels...")
    
    count_copied = 0
    
    for i, file in enumerate(files):
        try:
            # Read only the first row to check metadata (Fast)
            df = pd.read_csv(file, nrows=1)
            if df.empty: continue
            
            is_cargo = False
            
            # Check 1: VesselGroup column (Best/Easiest)
            if 'VesselGroup' in df.columns:
                group = str(df.iloc[0]['VesselGroup'])
                if 'Cargo' in group:
                    is_cargo = True
            
            # Check 2: VesselType Code (Fallback)
            if not is_cargo and 'VesselType' in df.columns:
                v_type = df.iloc[0]['VesselType']
                try:
                    if int(v_type) in CARGO_CODES:
                        is_cargo = True
                except: pass

            # Copy if valid
            if is_cargo:
                shutil.copy(file, os.path.join(OUTPUT_FOLDER, os.path.basename(file)))
                count_copied += 1
                
            if i % 1000 == 0:
                print(f"Processed {i}/{len(files)}... (Found {count_copied} Cargo)")

        except Exception as e:
            continue

    print(f"\n--- FILTER COMPLETE ---")
    print(f"Total Files Scanned: {len(files)}")
    print(f"Cargo Ships Isolated: {count_copied}")
    print(f"Location: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    filter_cargo()