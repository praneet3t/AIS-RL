import fiona
import pandas as pd
import os
from shapely.geometry import shape

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\AISVesselTracks2024\AISVesselTracks2024.gpkg"
OUTPUT_DIR = r"data/processed/probe_samples"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def probe_file():
    print(f"Opening GPKG: {INPUT_FILE}")
    
    # Open the file in "Stream Mode" (Low RAM)
    with fiona.open(INPUT_FILE, 'r') as source:
        # 1. Inspect Metadata Columns
        # This tells us EVERY variable available in the file
        schema = source.schema['properties']
        print("\n--- AVAILABLE METADATA COLUMNS ---")
        for key, dtype in schema.items():
            print(f" - {key} ({dtype})")
        print("----------------------------------\n")

        # 2. Extract First 5 Tracks
        print("Extracting first 5 tracks for inspection...")
        
        for i, feature in enumerate(source):
            if i >= 5: break # Stop after 5
            
            props = feature['properties']
            mmsi = props.get('MMSI', 'Unknown')
            
            # Extract Geometry (Lat/Lon points)
            geom = shape(feature['geometry'])
            
            if geom.geom_type == 'MultiLineString':
                coords = []
                for line in geom.geoms:
                    coords.extend(line.coords)
            elif geom.geom_type == 'LineString':
                coords = list(geom.coords)
            else:
                continue

            # Create DataFrame
            df = pd.DataFrame(coords, columns=['LON', 'LAT'])
            
            # Inject ALL Metadata
            for col_name in schema.keys():
                val = props.get(col_name)
                df[col_name] = val
            
            # Save to CSV
            filename = f"{mmsi}_segment_{i}.csv"
            save_path = os.path.join(OUTPUT_DIR, filename)
            df.to_csv(save_path, index=False)
            print(f"Saved: {filename}")

if __name__ == "__main__":
    probe_file()