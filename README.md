# Drug Information Assistant

A Streamlit chatbot that answers questions using uploaded drug documentation. The app extracts text from PDFs, creates local embeddings, retrieves relevant passages, and sends the grounded context to Groq for an answer.

## Prerequisites

- Python 3.10 or newer
- A Groq API key
- Git, if cloning the repository

## Setup

1. Clone the repository and open its folder:

   ```powershell
   git clone <repository-url>
   cd "Cortex 6-Drug Chatbot"
   ```
2. Create and activate a virtual environment.

   **Windows PowerShell:**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Create `.streamlit/secrets.toml` in the project root and add your Groq API key:

   ```toml
   API_KEY = "your-groq-api-key"
   ```

   Do not commit this file or share the key. It is excluded by `.gitignore`.

## Run the application

With the virtual environment activated, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open that URL in a browser.

## Use the chatbot

1. Click **Add source**.
2. Upload a drug documentation PDF.
3. Wait for the document to finish processing.
4. Ask a question about the uploaded document.
5. Review the retrieved evidence shown in the **Sources** panel.

The first run may take longer because the `all-MiniLM-L6-v2` embedding model is downloaded and cached locally. Uploaded PDFs and generated indexes are stored in `storage/`, which is intentionally not tracked by Git.

## Troubleshooting

- **`KeyError: 'API_KEY'`**: Confirm that `.streamlit/secrets.toml` exists in the project root and contains `API_KEY`.
- **`streamlit` or another module is not recognized**: Activate `.venv`, then run the install commands again.
- **PowerShell blocks activation**: Run PowerShell as your user and execute `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned`, then activate the environment again.
- **Model download or PDF processing fails**: Check your internet connection and confirm that the uploaded file is a readable PDF.
