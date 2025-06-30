
# Chat with Your PDFs using RAG

RAG stands for Retrieval-Augmented Generation. It’s a technique used in natural language processing (NLP) where a language model generates answers by combining:

🔹 Retrieval
It fetches relevant documents or text chunks from a knowledge base (like PDFs, websites, or databases) based on the user's query.

🔹 Augmented Generation
Then, the retrieved information is injected into the prompt of a language model (like GPT, LLaMA, or Groq LLM), which uses it to generate a grounded and accurate answer.

 Why use RAG?
LLMs (like GPT or LLaMA) have limited memory of external facts.

RAG extends their knowledge with real-time, external, and up-to-date information.

It reduces hallucination and increases accuracy for tasks like:

Question answering

Chat with documents

Legal or technical assistant tools
A Streamlit web app that lets you interactively chat with PDF documents using Retrieval-Augmented Generation (RAG), powered by:

- LangChain
- ChromaDB
- Groq LLMs (LLaMA 3)
- HuggingFace embeddings

---

## Features

- Upload one or multiple PDF files.
- Ask natural-language questions about their content.
- Uses LLM with retrieval and memory to generate context-aware answers.
- Chat history per session.


---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
````

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file with your HuggingFace token:

```
HF_TOKEN=your_huggingface_token
```

You will be prompted to enter your Groq API key in the app UI.

---

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser. Upload PDFs, select a model, enter your API key, and start chatting.

---

## Dependencies

Make sure to include at least the following in your `requirements.txt`:

```txt
streamlit
langchain
langchain-community
langchain-core
langchain-chroma
langchain-groq
langchain-huggingface
chromadb
python-dotenv
```


---

## Example Use Case

* Upload technical papers or documentation.
* Ask specific questions without reading the entire content.
* Use chat history to maintain conversation context.

---

## API Keys

* Groq API Key: Required to use the LLaMA 3 models via `langchain-groq`.
* HuggingFace Token: Needed for text embedding generation.

---

## Credits

Built using LangChain, Streamlit, and Groq.

---

## License

MIT License. See `LICENSE` file.


