# RAG App for Generating Topical Answers on YT Video Content

This project is a Retrieval-Augmented Generation (RAG) application that allows users to submit queries and receive relevant responses. The application is built using FastAPI for the backend and React.js with Typescript for the frontend.

## Project Structure

The project is organized into two main directories: `backend` and `frontend`.

### Backend

The backend is built with FastAPI and includes the following components:

- **main.py**: Entry point of the FastAPI application. Initializes the app and sets up middleware and routes.
- **api/endpoints.py**: Defines API endpoints for handling user queries.
- **models/rag_model.py**: Contains the implementation of the RAG model for processing queries and generating responses.
- **db/vector_db.py**: Manages the connection to the vector database for storing and retrieving vectors.
- **requirements.txt**: Lists the dependencies required for the backend.

### Frontend

The frontend is built with React.js and includes the following components:

- **App.tsx**: Main component that sets up routing and renders the layout, based on Material-UI in Typescript
- **pages/Home.tsx**: Home page that displays the menu to add videos, query against existing ones, and view the answer generated. It has several child components
- **components/TraceSidePanel.tsx**: The 'admin mode' that provides enriched information about past queries submitted


## Setup Instructions

### Backend

1. Navigate to the `backend` directory.
2. NOTE that you will need, if running with LLM-generated responses, an API key to access the GPT 4o-mini model preferred by this RAG. It is also possible to run without LLM integration, though this will result in the webapp simply displaying the most-releveant excerpts of YT video transcripts. Use `dotenv-template.txt` to make a `.env` on the toplevel that will be read by the FastAPI backend. If using LLM-less setup, specify the values for the section headed by `Toggle-on 1: openai for LLM`; if not, specify the values for the section headed by `Default: free/local`
3. Activate a virtual environment to isolate your library, and install the required dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   ```
4. Run the FastAPI application:
   ```
   uvicorn app.main:app --reload
   ```

### Frontend

1. Navigate to the `frontend` directory.
2. Install the required dependencies, using Node 18+:
   ```
   # nvm commands, if needed ...
   npm install
   ```
3. Start the React application:
   ```
   npm start
   ```

## Evaluation Framework

There is a password-walled admin-level inspection in the frontend that collects some metrics intended to signify the quality of the responses, and aggregated into a single score scaled from 0 to 1. The factors that comprise this aggregation (each factor itself is also similarly scaled 0 to 1, with higher numbers being better) include the level of similarity of the retrieved embeddings accroding to FAISS (Facebok AI Similarity Search), recency of the retrieved content, and diversity of the video excerpts used