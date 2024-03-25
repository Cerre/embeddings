
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

   [Download Large File](https://drive.google.com/yourfilelink)

   Place it in the appropriate directory as per your project structure.

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

## Running the Application

After installing the dependencies, you can run the application with the following command:

```bash
uvicorn main:app --reload
```

This will start the Uvicorn server with auto-reload enabled.

## Testing the API

Once the application is running, you can test the API by navigating to:

[http://localhost:8000/docs](http://localhost:8000/docs)

This will open the fastapi docs interface where you can try out different API endpoints.
