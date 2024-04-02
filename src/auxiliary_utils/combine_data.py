import pandas as pd
from pathlib import Path


def combine_csv_files(source_directory, target_directory, combined_filename):
    source_dir = Path(source_directory)
    target_dir = Path(target_directory)
    target_dir.mkdir(parents=True, exist_ok=True)  # Ensure target directory exists

    # Initialize an empty DataFrame to hold combined data
    combined_df = pd.DataFrame()

    # Loop through each CSV file in the source directory and append to the combined DataFrame
    for csv_file in source_dir.glob("*.csv"):
        df = pd.read_csv(csv_file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file in the target directory
    combined_path = target_dir / combined_filename
    combined_df.to_csv(combined_path, index=False)
    print(f"Combined CSV saved to {combined_path}")


# Usage
source_directory = "transcripts"
target_directory = "cleaned_data"
combined_filename = "combined_transcripts.csv"

combine_csv_files(source_directory, target_directory, combined_filename)
