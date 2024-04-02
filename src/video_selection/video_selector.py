import pandas as pd
from pytube import YouTube, Playlist


class VideoSelector:
    def __init__(self, single_video_url=None, playlist_url=None, url_list_path=None):
        self.single_video_url = single_video_url
        self.playlist_url = playlist_url
        self.url_list_path = url_list_path  # Store the path to the CSV file

    def get_videos(self):
        if self.single_video_url:
            yt = YouTube(self.single_video_url)
            return [yt]
        elif self.playlist_url:
            playlist = Playlist(self.playlist_url)
            return playlist.videos
        elif self.url_list_path:
            # Read the CSV file to get a list of video IDs
            video_df = pd.read_csv(self.url_list_path)
            video_ids = video_df[
                "videoId"
            ].tolist()  # Extract the 'videoId' column as a list
            # Create YouTube objects for each video ID
            videos = [
                YouTube(f"https://www.youtube.com/watch?v={video_id}")
                for video_id in video_ids
            ]
            return videos
        else:
            raise ValueError("No valid video or playlist URL provided.")
