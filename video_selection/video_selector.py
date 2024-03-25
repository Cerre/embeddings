from pytube import YouTube, Playlist

class VideoSelector:
    def __init__(self, single_video_url=None, playlist_url=None, url_list=None):
        self.single_video_url = single_video_url
        self.playlist_url = playlist_url
        self.url_list = url_list

    def get_videos(self):
        if self.single_video_url:
            yt = YouTube(self.single_video_url)
            return [yt]
        elif self.playlist_url:
            playlist = Playlist(self.playlist_url)
            return playlist.videos
        elif self.url_list:
            return [YouTube(url) for url in self.url_list]
        else:
            raise ValueError("No valid video or playlist URL provided.")
