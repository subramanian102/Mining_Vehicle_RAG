# FleetFix Pro: Automated Mining Truck Maintenance Engine

An enterprise-grade Retrieval-Augmented Generation (RAG) system engineered to parse dense, massive, and unstructured heavy industrial equipment documentation. This application processes complex technical manuals and uses language models to convert unstructured data into verifiable, step-by-step diagnostic workflows for on-site mechanics, significantly reducing vehicle repair downtime.

---

## Project Overview

In heavy industries like mining, vehicle downtime can generate significant financial losses per hour. Traditional maintenance manuals for large haul trucks span thousands of pages of dense text, troubleshooting codes, and schematic notes. When a mechanical breakdown occurs on-site, identifying the exact repair procedure manually is highly inefficient.

This application acts as an intelligent diagnostic assistant for workshop mechanics. It processes unstructured technical text, stores it within a semantic vector space, and utilizes large language models (LLMs) to retrieve precise, verifiable troubleshooting procedures. The system takes natural language inputs describing vehicle symptoms or system fault codes and returns structured, sequential diagnostic workflows. To eliminate hallucinations and ensure compliance with safety standards, it highlights the exact underlying context extracted from the technical manuals within an expandable drawer right inside the application workspace.

---

## Technical Architecture

The codebase utilizes a modular, detached pipeline structure that separates data ingestion from inference logic. This design ensures computational efficiency and horizontal scalability.

  ### 1. Data Ingestion and Semantic Indexing
The ingestion component converts raw text logs into a structured vector database.
* **Data Extraction**: The system reads raw structured manual fragments via the `datasets` library from a local `jsonl` repository format.
* **Text Chunking**: Technical documentation contains highly dense semantic parameters. To preserve context boundaries, text is broken into overlapping units using `RecursiveCharacterTextSplitter` with a strict `chunk_size` of 1000 characters and a `chunk_overlap` of 100 characters.
* **Vector Embeddings**: Text chunks are passed through a local Hugging Face embedding function (`all-MiniLM-L6-v2`) to transform textual statements into dense, high-dimensional semantic vectors.
* **Vector Store**: The computed vector tensors are written to a persistent instance of `Chroma`. This indexing step runs as a single pre-computation layer, saving significant computational runtime during chat interactions.

### 2. RAG Orchestration Layer
The core execution engine is orchestrated using LangChain Expression Language (LCEL) to construct an efficient data retrieval and generation loop.
* **Context Retrieval**: When a user inputs a search query, the system initializes a vector similarity search against Chroma, pulling the top 3 most relevant textual manual fragments based on cosine distance.
* **Prompt Structuring**: The retrieved chunks are injected into a strict system instruction prompt template. This model constraint forces the LLM to restrict its knowledge baseline exclusively to the provided documentation, explicitly preventing hallucinated text answers.
* **Dual-Model Core**: The orchestration framework features an interchangeable backend configuration router:
  * **Google Gemini 2.5 Flash**: Linked via secure API configurations to provide rapid, cloud-scaled reasoning capabilities.
  * **Meta-Llama 3.1 8B Instruct**: Configured via a local Hugging Face transformers execution pipeline, enabling on-site offline execution when internet access is unavailable in remote mining areas.

### 3. Dashboard Client Application
The front-end user interface is deployed as a single-page data application using Streamlit.
* **Persistent Conversational Tracking**: Implemented via Streamlit session state tokens to maintain structural chat records without clearing parameters between runtime component updates.
* **Dynamic Markdown Parsing**: The application processes the generated LLM text streams programmatically. If a response contains sequential diagnostic steps, the UI isolates them and renders them inside structured visual content dividers.
* **In-Window Reference Visibility**: To ensure high trust in safety-critical environments, the system displays an expandable drawer component beneath each response. This exposes the original source manuals and text segments that were used to build the answer.

---

## Project Structure

```text
├── src/
│   ├── ingestion.py   # Data extraction, chunking, and database persistence script
│   ├── rag_engine.py  # Core LangChain generation and LLM pipeline routing logic
│   └── app.py         # Streamlit application client UI and chat tracking environment
├── .env.example       # Template showing required environment variables
├── .gitignore         # Safety configurations ensuring no private tokens are leaked
└── requirements.txt   # Complete Python dependency installation framework
