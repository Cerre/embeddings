import pandas as pd
from datetime import datetime, timedelta
import re


def timestamp_to_datetime(timestamp):
    if not re.match(r"\d+:\d{2}", timestamp):
        return None
    try:
        return datetime.strptime(timestamp, "%H:%M:%S")
    except ValueError:
        return datetime.strptime(timestamp, "%M:%S")


def chunk_data(row, chunk_length_minutes=1, overlap_percentage=0.2):
    text = row["text"]
    try:
        lines = text.split("\n\n")[0]
        lines = lines.split("\n")
    except:
        print(
            f"Failed to gather lines for videoId: {row['videoId']} with name {row['title']}"
        )
        return []

    manuscript_index = next(
        (i for i, line in enumerate(lines) if "Manuskript" in line), -1
    )
    start_index = manuscript_index + 1 if manuscript_index != -1 else 0

    chunks = []
    valid_timestamps = []
    valid_texts = []
    current_text_accumulator = []
    lines = lines[start_index:]

    for idx, line in enumerate(lines):
        # Assuming row["title"] is a placeholder for the title check condition
        if line == row["title"]:
            break
        pattern = r"^\d{1,2}(:\d{2}){1,2}$"
        if re.match(pattern, line):
            if (
                current_text_accumulator
            ):  # If there's accumulated text, append it before resetting
                valid_texts.append(" ".join(current_text_accumulator))
                current_text_accumulator = []  # Reset accumulator
            valid_timestamps.append(timestamp_to_datetime(line))
        else:
            current_text_accumulator.append(line)

    # Append the last accumulated text (if any) after the loop ends
    if current_text_accumulator:
        valid_texts.append(" ".join(current_text_accumulator))

    chunk_length = timedelta(minutes=chunk_length_minutes)
    overlap_time = chunk_length * overlap_percentage

    i = 0
    print(len(valid_timestamps), len(valid_texts))
    if len(valid_timestamps) != len(valid_texts):
        valid_texts = valid_texts[1:]
    try:
        assert len(valid_timestamps) == len(valid_texts), print(
            f"Length of timestamps and texts do not match for videoId: {row['videoId']} with name {row['title']}"
        )
    except:
        breakpoint()
    while i < len(valid_timestamps):
        chunk_start_time = valid_timestamps[i]
        chunk_texts = []
        end_of_current_chunk = i
        while (
            i < len(valid_timestamps)
            and valid_timestamps[i] - chunk_start_time < chunk_length
        ):
            chunk_texts.append(valid_texts[i])
            i += 1
            end_of_current_chunk = i  # Update the end index of the current chunk

        # Save the current chunk
        if chunk_texts:
            formatted_start_time = chunk_start_time.strftime("%H:%M:%S")
            chunks.append((formatted_start_time, " ".join(chunk_texts)))

        # Adjust i for the overlap by finding the timestamp closest to but not past the overlap start time
        if i < len(valid_timestamps):
            # Ensure we have at least one timestamp to calculate overlap
            if end_of_current_chunk - 1 > 0:
                overlap_start_time = (
                    valid_timestamps[end_of_current_chunk - 1] - overlap_time
                )
                i = find_closest_timestamp_index(valid_timestamps, overlap_start_time)
                # print(i)
                # if i==340:
                #     breakpoint()
                # Find the index where the next chunk should start to maintain the overlap
            #     i = next((j for j, t in enumerate(valid_timestamps[:end_of_current_chunk]) if t >= overlap_start_time), end_of_current_chunk - 1)
            # else:
            #     # If we can't calculate overlap due to lack of timestamps, proceed with the next timestamp
            #     i = end_of_current_chunk

            # Ensure that 'i' is adjusted to always move forward by at least one position from the start of the current overlap calculation
            # i = max(i, end_of_current_chunk)

    return chunks


def find_closest_timestamp_index(valid_timestamps, target_time):
    min_diff = timedelta.max  # Initialize with maximum possible timedelta
    closest_index = -1  # Initialize with an invalid index
    for i, timestamp in enumerate(valid_timestamps):
        diff = abs(timestamp - target_time)  # Calculate absolute difference
        if diff < min_diff:
            min_diff = diff  # Update minimum difference
            closest_index = i  # Update closest index
    return closest_index


def process_row(row):
    print(row["videoId"], row["title"])
    chunks = chunk_data(row)
    # Convert each chunk tuple to a dictionary for easier readability
    chunk_dicts = [
        {"start_time": start_time, "chunk_text": chunk_text}
        for start_time, chunk_text in chunks
    ]
    return chunk_dicts


# Example usage
df = pd.read_csv("cleaned_data/combined_transcripts.csv")
df["text_chunks"] = df.apply(process_row, axis=1)
# Drop the 'text' column from the DataFrame
df_modified = df.drop(columns=["text"])

# Save the modified DataFrame to a JSON file
df_modified.to_json("combined_transcriptions_2.json", orient="records")
