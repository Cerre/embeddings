import pandas as pd
from datetime import datetime, timedelta


import pandas as pd
from datetime import datetime, timedelta
import re  # Regular expressions for validating timestamps


# Improved function to validate and convert timestamp strings to datetime objects
def timestamp_to_datetime(timestamp):
    # Validate timestamp format using regular expression
    if not re.match(r"\d{1,2}:\d{2}", timestamp):
        return None  # Return None or raise a custom exception for invalid timestamps
    try:
        return datetime.strptime(timestamp, "%H:%M:%S")
    except ValueError:
        # Handle cases where the timestamp might be in %M:%S format
        return datetime.strptime(timestamp, "%M:%S")


# Assuming we read the CSV file row by row
def process_csv_row(row):
    # Split the transcript text by lines
    lines = row["text"].split("\n")
    # Filter out lines that don't start with a timestamp
    lines = [
        line for line in lines if re.match(r"\d{1,2}:\d{2}", line.split(" ", 1)[0])
    ]
    # Now, you can chunk the filtered lines as previously described


# Function to chunk the data
def chunk_data(text, title, chunk_length_minutes=1, overlap_percentage=0.2):
    lines = text.split("\n")
    relevant_lines = [line for line in lines if line.strip() and not title in line]
    chunks = []
    chunk_start_time = None
    current_chunk = []
    chunk_length = timedelta(minutes=chunk_length_minutes)
    overlap = timedelta(minutes=chunk_length_minutes * overlap_percentage)

    for line in relevant_lines:
        parts = line.split(" ", 1)
        if len(parts) < 2:
            continue  # Skip lines that don't have a timestamp and text
        timestamp_str, text = parts
        timestamp = timestamp_to_datetime(timestamp_str.split("/")[0].strip())

        if chunk_start_time is None:
            chunk_start_time = timestamp

        if timestamp - chunk_start_time >= chunk_length:
            chunks.append((" ".join(current_chunk)))
            chunk_start_time += chunk_length - overlap
            current_chunk = [text]
        else:
            current_chunk.append(text)

    if current_chunk:
        chunks.append((" ".join(current_chunk)))

    return chunks


# Load the CSV file
input_csv_filename = (
    "cleaned_data/combined_transcripts.csv"  # Change this to your input CSV file path
)
df = pd.read_csv(input_csv_filename)


# Prepare a list to hold chunked data rows
chunked_rows = []

# Iterate over each row in the dataframe
for index, row in df.iterrows():
    process_csv_row(row)
    text_chunks = chunk_data(row["text"], title=row["title"])
    for i, chunk in enumerate(text_chunks):
        chunked_rows.append(
            {
                "title": row["title"],
                "chunk_text": chunk,
                "publishedAt": row["publishedAt"],
                "video_id": row["video_id"],
                "chunk_start_time": f"{i * (1 - 0.2):02}:00",  # Adjust the formula if needed
            }
        )

# Convert the list of chunked data rows to a new DataFrame
chunked_df = pd.DataFrame(chunked_rows)

# Save the chunked data to a new CSV file
output_csv_filename = "chunked_transcript.csv"
chunked_df.to_csv(output_csv_filename, index=False)

print(f"Data chunked and saved to {output_csv_filename}")
