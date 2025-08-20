# RAG App Frontend

This is the frontend part of the RAG (Retrieval-Augmented Generation) application built with React.js. The frontend communicates with the FastAPI backend to submit user queries and display responses.

## Getting Started

To get started with the frontend, follow these steps:

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd rag-app/frontend
   ```

2. **Install dependencies**:
   Make sure you have Node.js installed. Then run:
   ```
   npm install
   ```

3. **Run the application**:
   Start the development server with:
   ```
   npm start
   ```
   This will launch the application in your default web browser.

## Project Structure

The frontend project is structured as follows:

- `src/`: Contains the source code for the React application.
  - `App.tsx`: The main component that sets up routing and layout.
  - `components/`: Contains reusable components.
    - `QueryForm.tsx`: A component for submitting queries to the backend.
  - `pages/`: Contains different pages of the application.
    - `Home.tsx`: The home page displaying the query form and results.

## Usage

Once the application is running, you can submit queries through the form on the home page. The responses will be fetched from the backend and displayed accordingly.

## Evaluation

The quality of responses can be assessed using the evaluation framework implemented in the backend. Make sure to refer to the backend documentation for details on how to evaluate the responses.

## Contributing

If you would like to contribute to the project, please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.