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
2. Activate a virtual environment to isolate your library, and install the required dependencies:
   ```
   python3 -m venv venv
   source venv/bin/activate
   pip3 install -r requirements.txt
   ```
3. Run the FastAPI application:
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