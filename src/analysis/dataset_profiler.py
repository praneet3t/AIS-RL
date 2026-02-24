import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import Fullscreen
import os
import glob
import numpy as np

# --- CONFIGURATION ---
# WE USE THE ABSOLUTE PATH TO PREVENT "0 TRACKS" ERROR
# Check if this matches your actual folder structure
INPUT_FOLDER = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\src\data\data\processed\long_voyages"
OUTPUT_REPORT_DIR = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\outputs\reports"
OUTPUT_MAP_DIR = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\outputs\maps"

# Ensure output directories exist
os.makedirs(OUTPUT_REPORT_DIR, exist_ok=True)
os.makedirs(OUTPUT_MAP_DIR, exist_ok=True)

# Colors for the map
COLOR_MAP = {
    'Cargo': '#1f78b4',    # Blue
    'Tanker': '#e31a1c',   # Red
    'Fishing': '#ff7f00',  # Orange
    'Tug': '#6a3d9a',      # Purple
    'Passenger': '#33a02c',# Green
    'Pleasure': '#fdbf6f', # Light Orange
    'Other': '#b15928'     # Brown
}

def analyze_dataset():
    # 1. VERIFY FILES EXIST
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
    if not files:
        print(f"CRITICAL ERROR: No CSV files found in: {INPUT_FOLDER}")
        print("Please check if your extractor script saved them there.")
        return

    print(f"--- STARTING ANALYSIS OF {len(files)} TRACKS ---")
    
    summary_stats = []
    
    # --- PASS 1: SCANNING ---
    print("Pass 1: Harvesting Metadata...")
    
    for i, file in enumerate(files):
        try:
            # Read header + first row to be fast
            df = pd.read_csv(file)
            if len(df) < 2: continue
            
            # Metadata - Using the columns you confirmed exist
            mmsi = df.iloc[0].get('MMSI', 'Unknown')
            
            # USE VESSEL GROUP IF AVAILABLE (It is more accurate than Type ID)
            category = df.iloc[0].get('VesselGroup', 'Other')
            if pd.isna(category): category = 'Other'
            
            # Physics Stats
            dist = df.iloc[0].get('Calculated_Distance_Km', 0)
            avg_speed = df.iloc[0].get('Calculated_Speed_Knots', 0)
            
            # Check for physics columns (Turn/Heading)
            max_turn = df['Turn_Angle'].max() if 'Turn_Angle' in df.columns else 0
            avg_turn = df['Turn_Angle'].mean() if 'Turn_Angle' in df.columns else 0
            speed_var = df['Instant_Speed_Knots'].std() if 'Instant_Speed_Knots' in df.columns else 0

            summary_stats.append({
                'MMSI': mmsi,
                'Category': category,
                'Distance_Km': dist,
                'Avg_Speed_Knots': avg_speed,
                'Max_Turn_Angle': max_turn,
                'Avg_Turn_Angle': avg_turn,
                'Speed_Variance': speed_var,
                'FilePath': file
            })

            if i % 2000 == 0: print(f"Scanned {i}/{len(files)}...")

        except Exception as e:
            continue

    if not summary_stats:
        print("Error: Files found but no valid data extracted.")
        return

    df_master = pd.DataFrame(summary_stats)
    print(f"Successfully profiled {len(df_master)} valid tracks.")

    # --- GENERATE PLOTS ---
    print("Generating Analysis Plots...")
    sns.set_theme(style="whitegrid")

    # 1. Fleet Composition
    plt.figure(figsize=(10, 6))
    order = df_master['Category'].value_counts().index
    sns.countplot(data=df_master, x='Category', order=order, palette='viridis')
    plt.title(f"Fleet Composition (N={len(df_master)})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, "1_fleet_composition.png"))
    plt.close()

    # 2. Voyage Length Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df_master['Distance_Km'], bins=50, kde=True, color='navy')
    plt.title("Distribution of Voyage Distances (km)")
    plt.xlabel("Distance (km)")
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, "2_distance_dist.png"))
    plt.close()

    # 3. Speed Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df_master['Avg_Speed_Knots'], bins=30, kde=True, color='teal')
    plt.title("Speed Distribution (Knots)")
    plt.xlabel("Speed (knots)")
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, "3_speed_dist.png"))
    plt.close()

    # 4. Physics Check
    plt.figure(figsize=(10, 6))
    # Filter for cleaner plot
    subset = df_master.sample(min(2000, len(df_master)))
    sns.scatterplot(data=subset, x='Avg_Turn_Angle', y='Speed_Variance', hue='Category', alpha=0.5)
    plt.title("Physics Check: Turn Angle vs Speed Variance")
    plt.xlabel("Avg Turn Angle (Deg)")
    plt.ylabel("Speed Std Dev")
    plt.savefig(os.path.join(OUTPUT_REPORT_DIR, "4_physics_check.png"))
    plt.close()

    # --- PASS 2: MAPPING TOP 500 ---
    print("\nPass 2: Mapping Top 500 Longest Tracks...")
    
    top_500 = df_master.sort_values('Distance_Km', ascending=False).head(500)
    
    m = folium.Map(location=[30, -50], zoom_start=3, tiles="CartoDB positron")
    Fullscreen().add_to(m)

    count_mapped = 0
    for _, row in top_500.iterrows():
        try:
            track_df = pd.read_csv(row['FilePath'])
            
            # Downsample for browser performance
            step = max(1, int(len(track_df) / 200)) 
            track_small = track_df.iloc[::step]
            points = list(zip(track_small['LAT'], track_small['LON']))

            # Robust color picking
            cat_key = row['Category']
            # Handle cases like "Tanker" vs "Tankers" or "Cargo"
            color = 'gray'
            for key, c in COLOR_MAP.items():
                if key in str(cat_key):
                    color = c
                    break
            
            tooltip_txt = f"{row['Category']} | {row['Distance_Km']:.0f}km"
            popup_txt = f"""
            <div style="font-family:sans-serif; width:150px">
                <b>MMSI:</b> {row['MMSI']}<br>
                <b>Type:</b> {row['Category']}<br>
                <b>Dist:</b> {row['Distance_Km']:.1f} km<br>
                <b>Avg Spd:</b> {row['Avg_Speed_Knots']:.1f} kts
            </div>
            """

            folium.PolyLine(
                points, color=color, weight=2, opacity=0.6,
                tooltip=tooltip_txt, popup=folium.Popup(popup_txt, max_width=200)
            ).add_to(m)
            
            count_mapped += 1

        except Exception:
            continue

    map_path = os.path.join(OUTPUT_MAP_DIR, "top_500_long_haul.html")
    m.save(map_path)
    print(f"\nMap saved to: {map_path}")
    
    # Save Manifest
    df_master.to_csv(os.path.join(OUTPUT_REPORT_DIR, "dataset_manifest.csv"), index=False)
    print(f"Data Manifest saved to {OUTPUT_REPORT_DIR}")

if __name__ == "__main__":
    analyze_dataset()