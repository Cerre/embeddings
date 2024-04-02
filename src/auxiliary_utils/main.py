from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

# Replace 'YOUR_API_KEY' with your actual API key
api_key = "AIzaSyC6mfv2_WgxPXQsJhwV4wrRATmwa9dFgzs"
youtube = build("youtube", "v3", developerKey=api_key)


from googleapiclient.discovery import build


def get_channel_id_by_username(username):
    try:
        response = youtube.channels().list(forUsername=username, part="id").execute()

        # Check if 'items' key exists and has content
        if "items" in response and response["items"]:
            return response["items"][0]["id"]
        else:
            print(f"No channel found for username: {username}")
            return None
    except HttpError as e:
        print(f"An HTTP error occurred: {e.resp.status} - {e.content}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


def get_channel_videos(channel_id):
    content_details = (
        youtube.channels().list(id=channel_id, part="contentDetails").execute()
    )

    uploads_playlist_id = content_details["items"][0]["contentDetails"][
        "relatedPlaylists"
    ]["uploads"]

    videos = []
    next_page_token = None

    while True:
        playlist_items = (
            youtube.playlistItems()
            .list(
                playlistId=uploads_playlist_id,
                part="snippet",
                maxResults=50,
                pageToken=next_page_token,
            )
            .execute()
        )

        videos += [
            item["snippet"]["resourceId"]["videoId"] for item in playlist_items["items"]
        ]

        next_page_token = playlist_items.get("nextPageToken")
        if next_page_token is None:
            break

    return videos


import re


def sanitize_filename(text):
    """Sanitize a string to be safe for use as a filename."""
    text = re.sub(r'[\\/*?:"<>|]', "", text)  # Remove invalid file name characters
    return text[:200]  # Truncate to 200 characters to avoid too long filenames


def download_caption(video_id, lang="en"):
    try:
        # Fetch video details to get the title
        video_response = youtube.videos().list(id=video_id, part="snippet").execute()

        if not video_response["items"]:
            print(f"Video {video_id} not found.")
            return

        video_title = video_response["items"][0]["snippet"]["title"]
        sanitized_title = sanitize_filename(video_title)

        # Attempt to get the list of available captions
        captions_list = (
            youtube.captions().list(part="snippet", videoId=video_id).execute()
        )

        caption_available = False
        for item in captions_list["items"]:
            if item["snippet"]["language"] == lang:
                caption_available = True
                request = youtube.captions().download(id=item["id"], tfmt="srt")
                response = request.execute()

                # Include sanitized video title in the filename
                filename = f"captions/{sanitized_title}_{video_id}_{lang}.srt"
                with open(filename, "w") as file:
                    file.write(response)
                print(f"Saved caption to {filename}")
                save_checkpoint(checkpoint_file, video_id)
                break

        if not caption_available:
            print(f"No captions available in '{lang}' for video: {sanitized_title}")

    except HttpError as e:
        print(
            f"An HTTP error occurred for video ID {video_id}: {e.resp.status} - {e.content}"
        )
    except Exception as e:
        print(f"An unexpected error occurred for video ID {video_id}: {str(e)}")


import os


def load_checkpoint(filename):
    """Load processed video IDs from the checkpoint file."""
    try:
        with open(filename, "r") as file:
            processed_videos = file.read().splitlines()
        return set(processed_videos)
    except FileNotFoundError:
        return set()


def save_checkpoint(filename, video_id):
    """Save the processed video ID to the checkpoint file."""
    with open(filename, "a") as file:
        file.write(video_id + "\n")


# Example usage within your script
checkpoint_file = "checkpoint.txt"
processed_videos = load_checkpoint(checkpoint_file)

# channel_id = get_channel_id_by_username('JohnnyBoi_i')
channel_id = "UCldqb1GljWZzaRVtYXfvlAg"
channel_id = "UCMBATpFb--uLNAODOVWvCTA"  # Powerplay chess
video_ids = get_channel_videos(channel_id)

if channel_id:
    video_ids = get_channel_videos(channel_id)
    for idx, video_id in enumerate(video_ids):
        print(idx)
        download_caption(video_id)
        breakpoint()
else:
    print("Channel ID was not found. Please check the channel ID and try again.")
