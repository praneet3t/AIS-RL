import fiona
import pandas as pd
import os
import math
import numpy as np
from shapely.geometry import shape
from datetime import timedelta, datetime

# --- CONFIGURATION ---
INPUT_FILE = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\AISVesselTracks2024\AISVesselTracks2024.gpkg"
OUTPUT_DIR = r"data/processed/long_voyages"

# RL FILTERS
MIN_DISTANCE_KM = 500
MIN_SPEED_KNOTS = 5.0
MAX_SPEED_KNOTS = 30.0
MIN_DURATION_HOURS = 1.0

# PHYSICS PARAMETERS
# How much does turning slow you down?
# 0.0 = Constant speed (Linear)
# 5.0 = Heavy penalty (Sharp turns are 5x slower than straight lines)
TURN_PENALTY_FACTOR = 2.0 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates the bearing (heading) between two points in degrees."""
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    
    brng = math.atan2(y, x)
    return (math.degrees(brng) + 360) % 360

def haversine_distance(coords):
    """
    Calculates distance and geometry stats (Heading, Turn Angle).
    """
    R = 6371  # Earth radius km
    segments = []
    total_km = 0
    
    # We need at least 3 points to calculate a turn angle for the middle segment
    # For the first segment, we assume 0 turn.
    prev_bearing = None

    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i+1]
        
        # 1. Distance
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        dist = R * c
        total_km += dist
        
        # 2. Heading (COG)
        bearing = calculate_bearing(lat1, lon1, lat2, lon2)
        
        # 3. Turn Angle (Change in heading from previous segment)
        turn_angle = 0.0
        if prev_bearing is not None:
            # Calculate shortest difference (e.g., 350 -> 10 is 20 degrees, not 340)
            diff = abs(bearing - prev_bearing)
            turn_angle = min(diff, 360 - diff)
        
        segments.append({
            'dist_km': dist,
            'bearing': bearing,
            'turn_angle': turn_angle
        })
        
        prev_bearing = bearing
        
    return total_km, segments

def extract_dataset():
    print(f"--- STARTING PHYSICS-BASED EXTRACTION ---")
    print(f"Turn Penalty Factor: {TURN_PENALTY_FACTOR}")
    
    stats = {"scanned": 0, "saved": 0, "errors": 0}
    
    with fiona.open(INPUT_FILE, 'r') as source:
        # Get All Metadata Columns
        meta_schema = list(source.schema['properties'].keys())
        
        for feature in source:
            stats["scanned"] += 1
            if stats["scanned"] % 5000 == 0:
                print(f"Scanned {stats['scanned']}... Saved: {stats['saved']}")

            try:
                # 1. Geometry
                geom = shape(feature['geometry'])
                if geom.is_empty: continue
                
                if geom.geom_type == 'MultiLineString':
                    coords = []
                    for line in geom.geoms: coords.extend(line.coords)
                elif geom.geom_type == 'LineString':
                    coords = list(geom.coords)
                else: continue
                
                if len(coords) < 2: continue

                # 2. Calculate Geometry Stats
                total_dist_km, segments = haversine_distance(coords)
                
                if total_dist_km < MIN_DISTANCE_KM: continue

                # 3. Validate Time
                props = feature['properties']
                duration_mins = props.get('DurationMinutes', 0)
                if not duration_mins or duration_mins <= (MIN_DURATION_HOURS * 60): continue

                # Global Average Check (Filter bad data)
                avg_speed_global = (total_dist_km * 0.539957) / (duration_mins / 60.0)
                if avg_speed_global < MIN_SPEED_KNOTS or avg_speed_global > MAX_SPEED_KNOTS:
                    continue

                # 4. PHYSICS-BASED TIME INTERPOLATION
                # We distribute the total duration based on "Effort"
                # Effort = Distance * (1 + Penalty * Turn_Angle)
                # Sharp turns = More Effort = More Time = Slower Speed
                
                total_effort = 0
                for seg in segments:
                    # Angle is in degrees. Normalize 180 turn = 1.0 penalty scale
                    angle_penalty = (seg['turn_angle'] / 180.0) * TURN_PENALTY_FACTOR
                    weight = seg['dist_km'] * (1 + angle_penalty)
                    seg['weight'] = weight
                    total_effort += weight
                
                # Distribute Time
                start_time_str = props.get('TrackStartTime')
                try:
                    if isinstance(start_time_str, str):
                        start_time = pd.to_datetime(start_time_str.replace('Z', ''))
                    else: continue
                except: continue

                timestamps = [start_time]
                current_time = start_time
                segment_speeds = [] # Instantaneous speeds

                for seg in segments:
                    # Fraction of total time allocated to this segment
                    time_fraction = seg['weight'] / total_effort
                    seg_duration_mins = duration_mins * time_fraction
                    
                    # Advance time
                    current_time += timedelta(minutes=seg_duration_mins)
                    timestamps.append(current_time)
                    
                    # Calculate Instant Speed for this segment
                    # Speed = Dist / Time
                    if seg_duration_mins > 0:
                        knots = (seg['dist_km'] * 0.539957) / (seg_duration_mins / 60.0)
                        segment_speeds.append(knots)
                    else:
                        segment_speeds.append(0)

                # 5. Build DataFrame
                df = pd.DataFrame(coords, columns=['LON', 'LAT'])
                
                # Columns lengths must match
                df['Estimated_Timestamp'] = timestamps[:len(df)]
                
                # Segment data corresponds to the interval between points
                # We'll assign segment data to the "Start Point" of the segment
                # The last point has no "next segment", so we pad with NaN or 0
                
                # Pad lists to match df length
                bearings = [s['bearing'] for s in segments] + [0]
                turns = [s['turn_angle'] for s in segments] + [0]
                speeds = segment_speeds + [0] # Last point speed is 0 (arrived)

                df['COG_Heading'] = bearings
                df['Turn_Angle'] = turns
                df['Instant_Speed_Knots'] = [round(x, 2) for x in speeds]
                
                # Add Metadata
                for col in meta_schema:
                    df[col] = props.get(col)

                # Save
                mmsi = props.get('MMSI', 'Unknown')
                filename = f"{mmsi}_trip_{stats['scanned']}.csv"
                df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
                stats["saved"] += 1

            except Exception as e:
                stats["errors"] += 1
                continue

    print(f"\n--- DONE ---")
    print(f"Saved {stats['saved']} physics-enhanced tracks.")

if __name__ == "__main__":
    extract_dataset()