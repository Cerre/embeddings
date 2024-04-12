import os
from pytube import YouTube
from pytube.exceptions import PytubeError


class Downloader:
    def __init__(self, download_path="downloaded_audio"):
        self.download_path = download_path
        os.makedirs(self.download_path, exist_ok=True)

    async def download_audio(self, video):
        try:
            audio_streams = video.streams.filter(only_audio=True).first()
            if not audio_streams:
                print(f"No audio stream found for video: {video.title}")
                return None

            audio_file_path = os.path.join(self.download_path, f"{video.video_id}.mp4")
            audio_streams.download(
                output_path=self.download_path, filename=f"{video.video_id}.mp4"
            )
            return audio_file_path
        except PytubeError as e:
            print(f"An error occurred while downloading the video {video.title}: {e}")
            return None
        except Exception as e:
            print(
                f"An unexpected error occurred while processing the video {video.title}: {e}"
            )
            return None
