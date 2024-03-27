
# Embeddings

Creating embeddings from transcriptions to use for RAG and llm 


## Installation

Follow these steps to get your development environment set up:

1. **Clone the repository (optional):**

   ```bash
   git clone git@github.com:Cerre/embeddings.git
   cd embeddings
   ```

2. **Download the large file:**

   Download the required large file from Google Drive:

   [Download Large File](https://drive.google.com/file/d/1MzT5x84FK-TX6UQfFHL2Bcy0GprK-rnw/view?usp=drive_link)

    ```bash
   mkdir data/
    ```
    Place file in data

3. **Set up a virtual environment:**

   For Unix/macOS:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   For Windows:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Set up openai api key:**

   ```bash
   export OPENAI_API_KEY=YOUR_API_KEY
   ```

## Running and testing the Api


```bash
uvicorn main:app --reload
```

This will start the Uvicorn server with auto-reload enabled.

Once the application is running, you can test the API by navigating to:

[http://localhost:8000/docs](http://localhost:8000/docs)

This will open the fastapi docs interface where you can try out different API endpoints.

## Download and transcribe audio from youtube

```bash
python main_youtube_transcriber
```