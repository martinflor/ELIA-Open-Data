import pandas as pd
from bid_acceptance_model import build_capacity_periods, generate_slide2_3_4_examples
import os

# Try to load from CSV if it exists, otherwise fetch from API
csv_path = "ods125.csv"
if os.path.exists(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
else:
    print("CSV file not found. Fetching data from ELIA API...")
    import elia
    # Fetch data for the date range that includes the example block
    df = elia.fetch_afrr_capacity_range_all_chunked("2025-01-01", "2025-01-11")
    if df is None or df.empty:
        raise ValueError("No data fetched from API. Please provide ods125.csv file.")
    print(f"Fetched {len(df)} rows from API")

# Build 4-hour period index (required by your model)
print("Building capacity periods...")
cap_df = build_capacity_periods(df)
print(f"Built {len(cap_df)} capacity periods")

# Generate plots for the same block used in your slides
print("Generating HTML plots...")
try:
    paths = generate_slide2_3_4_examples(
        cap_df=cap_df,
        direction="Up",
        block_start="2025-01-10 04:00:00",
        out_dir="plotly_exports"
    )
    
    print("\n" + "="*60)
    print("HTML plots generated successfully!")
    print("="*60)
    for i, path in enumerate(paths, 1):
        if os.path.exists(path):
            file_size = os.path.getsize(path) / 1024  # KB
            print(f"Slide {i+1}: {path} ({file_size:.1f} KB)")
        else:
            print(f"Slide {i+1}: {path} (FILE NOT FOUND!)")
    print("="*60)
    print(f"\nOpen the HTML files in your browser to view the interactive plots.")
    
except Exception as e:
    print(f"\nError generating plots: {e}")
    import traceback
    traceback.print_exc()
