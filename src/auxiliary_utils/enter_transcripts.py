import csv
from pathlib import Path


# Function to read the original CSV file
def read_original_csv(file_path):
    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)  # Convert to list for easier index access


# Function to write a new CSV file for each input
def write_new_csv(row, input_text, directory):
    headers = row.keys() | {"text"}  # Union of dict keys and new column
    file_name = f"{directory}/{row['videoId']}.csv"
    with open(file_name, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        row["text"] = input_text  # Add the new text to the row
        writer.writerow(row)


# Function to capture multi-line input
def get_multi_line_input(row, end_command="END"):
    print(
        f"Enter/paste your content for video:  {row['title']}. Type 'END' (without quotes) on a new line to finish."
    )
    lines = []
    while True:
        line = input()
        if line.strip() == end_command:
            break
        lines.append(line)
    return "\n".join(lines)


# Function to find already processed videoIds
def find_processed_videos(directory):
    processed = set()
    for file in directory.glob("*.csv"):
        processed.add(file.stem)  # file.stem gives the filename without extension
    return processed


def main():
    # Adjust the directory as needed
    directory = Path("transcripts")  # Current working directory
    original_csv_path = "channel_videos.csv"
    rows = read_original_csv(original_csv_path)
    processed_video_ids = find_processed_videos(directory)

    for index, row in enumerate(rows):
        if row["videoId"] in processed_video_ids:
            continue  # Skip already processed videos

        # Get multi-line user input from terminal
        input_text = get_multi_line_input(row)
        write_new_csv(row, input_text, directory)
        print(f"Saved file for videoId: {row['videoId']} with added text.")

        # Break after processing all rows or continue for each new input
        if index == len(rows) - 1:
            print("All rows processed.")
            break


if __name__ == "__main__":
    main()
