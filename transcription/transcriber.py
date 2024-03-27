import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from faster_whisper import WhisperModel
import whisperx
import time


from abc import ABC, abstractmethod
import asyncio
from pathlib import Path
from datetime import datetime


class Transcriber(ABC):
    def __init__(
        self,
        model_name="large-v3",
        device="cpu",
        output_dir="transcripts",
        compute_type="auto",
    ):
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.compute_type = compute_type
        self.model = self.load_model()  # Load model on initialization

    @abstractmethod
    def load_model(self):
        """
        Load and return the model.
        This method must be implemented by each subclass.
        """
        pass

    @abstractmethod
    async def transcribe_audio(self, audio_file_path):
        """
        Transcribe audio file. This method must be implemented by each subclass.
        """
        pass


from faster_whisper import WhisperModel


class WhisperTranscriber(Transcriber):
    def load_model(self):
        return WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type
        )

    async def transcribe_audio(self, audio_file_path):
        start_time = datetime.now()
        segments, info = await asyncio.to_thread(
            self.model.transcribe,
            audio_file_path,
            beam_size=10,
            vad_filter=True,
            language="en",
        )
        segments_list = list(segments)
        transcript_text = " ".join([segment.text for segment in segments_list])
        metadata_dicts = [
            {
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text,
                "avg_logprob": round(segment.avg_logprob, 2),
            }
            for segment in segments_list
        ]

        execution_time = datetime.now() - start_time
        print(
            f"Transcription and processing completed in {execution_time}. Model used: {self.model_name}"
        )

        return transcript_text, metadata_dicts


class WhisperxTranscriber(Transcriber):
    def __init__(
        self,
        model_name="large-v3",
        device="cpu",
        output_dir="transcripts",
        compute_type="auto",
        batch_size=1,
    ):
        super().__init__(model_name, device, output_dir, compute_type)
        self.batch_size = batch_size  # Specific to WhisperxTranscriber

    def load_model(self):
        return whisperx.load_model(
            self.model_name, device=self.device, compute_type=self.compute_type
        )

    async def transcribe_audio(self, audio_file):
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size, language="en")
        result_aligned = self._align_result(result, audio)
        transcript_text = " ".join(
            [segment["text"] for segment in result_aligned["segments"]]
        )
        return transcript_text, result_aligned["segments"]

    def _align_result(self, result, audio):
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=self.device
        )
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )
        return result_aligned

    def diarize_audio(self, max_speakers=2):
        hf_token = os.getenv("HF_TOKEN")
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token, device=self.device, max_speakers=max_speakers
        )
        diarize_segments = diarize_model(audio)
        result_diarization = whisperx.assign_word_speakers(diarize_segments, result)
