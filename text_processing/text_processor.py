import re
import spacy
from pathlib import Path
import pandas as pd

class TextProcessor:
    def __init__(self, use_spacy=True, output_dir="processed_texts"):
        self.use_spacy = use_spacy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.nlp = self.download_spacy_model()

    def sophisticated_sentence_splitter(self, text):
        text = self.remove_pagination_breaks(text)
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences    

    def download_spacy_model(self, model_name="en_core_web_sm"):
        try:
            return spacy.load(model_name) # Load the model if already installed
        except OSError: # If not installed, download it
            print(f"Downloading spaCy model {model_name}...")
            spacy.cli.download(model_name)
            return spacy.load(model_name)



    def remove_pagination_breaks(self, text: str) -> str:
        text = re.sub(r'-(\n)(?=[a-z])', '', text) # Remove hyphens at the end of lines when the word continues on the next line
        text = re.sub(r'(?<=\w)(?<![.?!-]|\d)\n(?![\nA-Z])', ' ', text) # Replace line breaks that are not preceded by punctuation or list markers and not followed by an uppercase letter or another line break   
        return text

    @staticmethod
    def clean_filename(title):
        title = re.sub('[^\w\s-]', '', title)
        return re.sub('[-\s]+', '_', title).strip().lower()


    def save_processed_data(self, audio_file_name, sentences, metadata_dicts):
        # Save sentences to a text file
        text_file_path = self.output_dir / f"{audio_file_name}_sentences.txt"
        with open(text_file_path, "w") as file:
            for sentence in sentences:
                file.write(sentence + "\n")

        # Save metadata to CSV and JSON
        metadata_csv_path = self.output_dir / f"{audio_file_name}_metadata.csv"
        metadata_json_path = self.output_dir / f"{audio_file_name}_metadata.json"
        df = pd.DataFrame(metadata_dicts)
        df.to_csv(metadata_csv_path, index=False)
        df.to_json(metadata_json_path, orient='records', indent=4)

        print(f"Output files saved: {text_file_path}, {metadata_csv_path}, {metadata_json_path}")