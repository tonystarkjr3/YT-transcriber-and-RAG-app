# RAG App Backend

This is the backend component of the RAG (Retrieval-Augmented Generation) application, built using FastAPI. The backend handles user queries, processing them with a RAG model, and returning relevant responses.

## Project Structure

The backend is organized into the following directories and files:

- `app/`: Contains the main application code.
  - `main.py`: Entry point of the FastAPI application.
  - `api/`: Contains API endpoint definitions.
    - `endpoints.py`: Handles user queries and responses.
  - `models/`: Contains the RAG model implementation.
    - `rag_model.py`: Loads the model and processes queries.
  - `db/`: Manages the vector database connection.
    - `vector_db.py`: Functions for storing and retrieving vectors, as well as enriched information accessible through the Admin-mode UIs

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd rag-app/backend
   ```

2. **Create a virtual environment:**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Run the FastAPI application:**
   ```
   uvicorn app.main:app --reload
   ```

   The application will be available at `http://127.0.0.1:8000`.

## Usage

- The API endpoints can be accessed at `/api` (defined in `endpoints.py`).
- You can submit queries to the RAG model and receive responses based on the provided corpus.
- The highlight functions exposed as routes in a REST interface are
  - `HTTP POST` to `api/videos/add` that will locate a youtube video at a specified link, perform the embedding of it into a vectorized format, and store both the natural language of the excerpt and the embedding inside a customizable SQL DB and an embedding vector retrievably speedily through FAISS (Facebook AI Similarity Search)
  - `HTTP POST` to `api/query` that contains the question of the user, transformed into a dense embedding compared with stored vectors; these aim to select the most relevant encoded transcript excerpts
  - `HTTP GET` to `api/admin/traces` & `api/admin/trace/<trace-id>` that produces the metadata related to historical query events: the top-k selected excerpts (linked through the IDs on the vectors), the timestamp in the video at which they occurred, the prompt built out of the vectors and subsequently sent to LLM, and a quantification of the 'confidence' of the answer (a formula yielding a number that represents the forecasted quality of the response based on video diversity, match strength based on FAISS, etc.)