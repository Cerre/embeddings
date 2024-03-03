from googleapiclient.discovery import build
import csv
import os

# Replace 'YOUR_API_KEY' with your actual API key
api_key = 'AIzaSyC6mfv2_WgxPXQsJhwV4wrRATmwa9dFgzs'
youtube = build('youtube', 'v3', developerKey=api_key)

def get_channel_videos(channel_id):
    uploads_playlist_id = youtube.channels().list(
        id=channel_id,
        part='contentDetails'
    ).execute()['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    
    next_page_token = None
    video_count = 0

    while True:
        print(video_count)
        playlist_items = youtube.playlistItems().list(
            playlistId=uploads_playlist_id,
            part='snippet',
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        for item in playlist_items['items']:
            video = {
                'title': item['snippet']['title'],
                'publishedAt': item['snippet']['publishedAt'],
                'videoId': item['snippet']['resourceId']['videoId']
            }
            save_video_to_csv(video)  # Save each video immediately
            video_count += 1

        next_page_token = playlist_items.get('nextPageToken')
        if next_page_token is None:
            break

    return video_count


def save_video_to_csv(video, filename='channel_videos.csv'):
    # Checks if the file already exists to decide on writing the header
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        fieldnames = ['title', 'publishedAt', 'videoId']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only if file didn't exist

        writer.writerow(video)


# Example usage
channel_id = "UCMBATpFb--uLNAODOVWvCTA"  # Example: Powerplay Chess
channel_id = "UCldqb1GljWZzaRVtYXfvlAg"
videos = get_channel_videos(channel_id)

print(f"Saved {len(videos)} videos to channel_videos.csv")
