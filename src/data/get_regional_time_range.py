import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Check the MAIN folder, not the dense one
SOURCE_FOLDER = r"C:\Users\apran\Videos\Cin\LIBRARY\AIS-RL\data\processed\cargo_dense_region"

def audit_full_year():
    files = glob.glob(os.path.join(SOURCE_FOLDER, "*.csv"))
    print(f"Auditing {len(files)} tracks from the main folder...")

    dates = []
    # Sample 20% of files to get a representative spread
    for i in range(0, len(files), 5):
        try:
            # Just read the first timestamp of each file
            df = pd.read_csv(files[i], usecols=['Estimated_Timestamp'], nrows=1)
            dates.append(pd.to_datetime(df['Estimated_Timestamp'].iloc[0], utc=True))
        except:
            continue

    date_series = pd.Series(dates)
    
    print("\n" + "="*40)
    print("      DATASET TEMPORAL AUDIT")
    print("="*40)
    print(f"Earliest: {date_series.min()}")
    print(f"Latest:   {date_series.max()}")
    print(f"Total Unique Months Found: {date_series.dt.month.nunique()}")
    
    # Show a small bar chart of data per month
    date_series.dt.month.value_counts().sort_index().plot(kind='bar')
    plt.title("Tracks per Month (2024)")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    audit_full_year()