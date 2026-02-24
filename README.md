# Maritime RL Agent

RL-based ship routing using AIS data from the Great Lakes region.

## Data Coverage
- **Region**: Great Lakes (North America)
- **Bounding Box**: 
  - Longitude: -92.16° to -74.39° W
  - Latitude: 41.45° to 48.73° N
- **Tracks**: 2,840 vessel trajectories
- **Vessel Type**: Cargo ships

## Structure
```
AIS-RL/
├── data/
│   ├── enriched_tracks_final/  # 2,840 AIS tracks with weather
│   └── weather_data/           # Weather data (NetCDF + CSV)
├── src/
│   ├── analysis/               # Data analysis scripts
│   ├── data/                   # Data processing scripts
│   └── utils/                  # Utility functions
├── outputs/                    # Generated outputs
├── configs/                    # Configuration files
└── all_tracks.png             # Visualization of all tracks
```
