import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
CSV_FOLDER = "/workspace/AIS-RL/data/enriched_tracks_final/*.csv"
FRAMES_PER_DAY = 2  # Higher = Slower animation. 2 per day = ~730 frames for the year.
TAIL_LENGTH_DAYS = 14 # How long the "comet trail" of ships lasts. Longer = more crowded look.

print("🚢 Starting massive data load for 2024 visualization...")

# ==========================================
# 2. LOAD AND FLATTEN THE DATA
# ==========================================
csv_files = glob.glob(CSV_FOLDER)
all_points = []

print(f"Processing {len(csv_files)} files. This might take a minute...")
for file in tqdm(csv_files):
    # Only load needed columns to save RAM
    try:
        df = pd.read_csv(file, usecols=['LAT', 'LON', 'Estimated_Timestamp', 'MMSI']).dropna()
        if len(df) > 5:
            all_points.append(df)
    except Exception as e:
        continue # Skip corrupted files

# Combine into master timeline
master_df = pd.concat(all_points, ignore_index=True)

print("🕰️ Standardizing timestamps...")
# Using format='mixed' and utc=True to handle messy AIS times without crashing
master_df['time'] = pd.to_datetime(master_df['Estimated_Timestamp'], format='mixed', utc=True)

# Filter strictly for 2024 and sort
master_df = master_df[master_df['time'].dt.year == 2024]
master_df = master_df.sample(frac=0.1, random_state=42).sort_values('time') # Optional: downsample to 10% for faster rendering

print(f"✅ Data loaded: {len(master_df):,} total AIS pings spanning 2024.")

# Get bounds for the map view
min_lon, max_lon = master_df['LON'].min(), master_df['LON'].max()
min_lat, max_lat = master_df['LAT'].min(), master_df['LAT'].max()

# ==========================================
# 3. PREPARE THE BASE MAP (LAND)
# ==========================================
print("🗺️ Preparing geopolitical base map...")
try:
    # Load built-in low-res world boundaries
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Create a bounding box polygon to clip the world map to our region
    bounding_box = box(min_lon - 1, min_lat - 1, max_lon + 1, max_lat + 1)
    local_map = world.clip(bounding_box)
except Exception as e:
    print("Warning: Could not load geopandas map data. Falling back to black background.")
    local_map = None

# ==========================================
# 4. BUILD THE VISUAL CANVAS
# ==========================================
# Dark theme setup
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(16, 9)) # 16:9 Cinematic aspect ratio
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set_xticks([]); ax.set_yticks([]) 
ax.set_xlim(min_lon, max_lon)
ax.set_ylim(min_lat, max_lat)

# A. Plot Land Layer (if available)
if local_map is not None:
    # Plot land as dark grey, ocean stays black
    local_map.plot(ax=ax, color='#333333', edgecolor='#1a1a1a', linewidth=0.5, zorder=1)
else:
    ax.set_facecolor('black')

# B. Plot Ship Scatter Layer (Starts empty)
# Using a glowing cyan/blue color pallete
scatter = ax.scatter([], [], s=0.5, c='#00f7ff', alpha=0.4, edgecolors='none', zorder=2)

# C. Plot Date Text Layer
date_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white', 
                     fontsize=20, ha='left', fontweight='bold', alpha=0.9, zorder=3)
stats_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, color='#cccccc',
                     fontsize=12, ha='left', zorder=3)

# ==========================================
# 5. THE SLOW-BURN ANIMATION LOOP
# ==========================================
start_time = master_df['time'].min()
end_time = master_df['time'].max()
days_total = (end_time - start_time).days
num_frames = int(days_total * FRAMES_PER_DAY) # e.g., 365 * 2 = 730 frames

print(f"🎬 Preparing to render {num_frames} frames for a slow, progressive visualization.")
time_steps = pd.date_range(start=start_time, end=end_time, periods=num_frames)

def update(frame_idx):
    current_time = time_steps[frame_idx]
    
    # Define the rolling window (e.g., past 14 days up to current moment)
    window_start = current_time - pd.Timedelta(days=TAIL_LENGTH_DAYS)
    
    # Fast boolean indexing to grab the slice
    mask = (master_df['time'] > window_start) & (master_df['time'] <= current_time)
    current_data = master_df[mask]
    
    # Update ships
    if not current_data.empty:
        scatter.set_offsets(np.c_[current_data['LON'], current_data['LAT']])
        # Optional: vary opacity by age so older pings fade out
        # ages = (current_time - current_data['time']).dt.total_seconds()
        # alphas = 1 - (ages / (TAIL_LENGTH_DAYS * 24 * 3600))
        # scatter.set_alpha(np.clip(alphas, 0.1, 0.8))
    else:
        scatter.set_offsets(np.empty((0, 2)))
        
    # Update text
    date_text.set_text(current_time.strftime('%B %d, %Y'))
    stats_text.set_text(f"Active Vessels (14-day window): {len(current_data['MMSI'].unique()):,}")
    
    return scatter, date_text, stats_text

# Animate
# interval=40 means 40ms per frame = 25 frames per second. 
# 730 frames / 25 fps = ~30 second animation duration.
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=40, blit=True)

print("☕ Rendering GIF. This will take several minutes due to high frame count and map layers...")
# Using a higher DPI for better quality on big screens
ani.save('slow_map_timelapse.gif', writer='pillow', fps=25, dpi=150, savefig_kwargs={'facecolor': 'black'})
print("✨ DONE! 'slow_map_timelapse.gif' is ready.")