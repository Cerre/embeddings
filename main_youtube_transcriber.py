import asyncio
from video_selection.video_selector import VideoSelector
from downloads.downloader import Downloader
from transcription.transcriber import Transcriber
from text_processing.text_processor import TextProcessor
import os
from numba import cuda

async def process_video(video, downloader, transcriber, text_processor):
    audio_path = await downloader.download_audio(video)
    if audio_path:  # Ensure audio was downloaded successfully
        audio_file_name = video.video_id  # Or any other naming convention you prefer
        transcript, metadata_dicts = await transcriber.transcribe_audio(audio_path, audio_file_name)
        sentences = text_processor.sophisticated_sentence_splitter(transcript)
        text_processor.save_processed_data(audio_file_name, sentences, metadata_dicts)
        print(f"Transcript: {transcript}\n\n")
        print(f"metdata_dicts: {metadata_dicts}\n\n")
        print(f"sentences: {sentences}")
    else:
        print(f"Failed to download audio for video: {video.title}")





def setup_cuda():
    cuda_toolkit_path = _get_cuda_toolkit_path()
    if cuda_toolkit_path:
        os.environ["PATH"] += os.pathsep + cuda_toolkit_path
        os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:/usr/local/cuda-12.3/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
    if cuda.is_available():
        print("CUDA is available. Using GPU for transcription.")
        device = "cuda"
        compute_type = "float16"  # Use FP16 for faster computation on GPU
    else:
        print("CUDA not available. Using CPU for transcription.")
        device = "cpu"
        compute_type = "auto"  # Use default compute type for CPU
    return device, compute_type

def _get_cuda_toolkit_path():
        # Common paths for CUDA Toolkit in WSL2/Linux
        possible_paths = [
            "/usr/local/cuda-12.3/bin", 
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path  # Return the first match
        return None  # Return None if no path found





async def main():
    # video_selector = VideoSelector(single_video_url="https://www.youtube.com/watch?v=PC8iEn8bdl8")
    video_selector = VideoSelector(single_video_url="https://www.youtube.com/watch?v=beAvFHP4wDI")
    
    videos = video_selector.get_videos()
    device, compute_type = setup_cuda()
    
    downloader = Downloader()
    transcriber = Transcriber(model_name = "tiny.en", device=device, output_dir="transcripts", compute_type=compute_type)  # Specify output directory for transcripts and metadata
    text_processor = TextProcessor()  # Assuming spacy is used for text processing

    tasks = [process_video(video, downloader, transcriber, text_processor) for video in videos]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
