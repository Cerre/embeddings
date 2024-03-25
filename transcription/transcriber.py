import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel

class Transcriber:
    def __init__(self, model_name="large-v3", device="cpu", output_dir="transcripts", compute_type="auto"):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compute_type = compute_type

    async def transcribe_audio(self, audio_file_path, audio_file_name):
        model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type)
        start_time = datetime.now()
        segments, _ = await asyncio.to_thread(model.transcribe, audio_file_path, beam_size=10, vad_filter=True, language="en")
        segments_list = list(segments)
        transcript_text = " ".join([segment.text for segment in segments_list])
        metadata_dicts = [{
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text,
            "avg_logprob": round(segment.avg_logprob, 2)
        } for segment in segments_list]

        execution_time = datetime.now() - start_time
        print(f"Transcription and processing completed in {execution_time}. Model used: {self.model_name}")

        return transcript_text, metadata_dicts

