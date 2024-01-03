# pdfreader_openai
# Description
Pdfreader_openai is an efficient tool for indexing and searching PDF text data using OpenAI APIs and FAISS (Facebook AI Similarity Search). This software is designed for rapid information retrieval and superior search accuracy.
# Libraries Used

Textract - A Python library for extracting text from any document.
Transformers - A library by Hugging Face providing state-of-the-art general-purpose architectures for Natural Language Understanding (NLU) and Natural Language Generation (NLG).
Langchain - A text processing and embeddings library.
FAISS (Facebook AI Similarity Search) - A library for efficient similarity search and clustering of dense vectors.

# Installing Dependencies

You can install all dependencies by running the following command:

   pip install langchain openai textract transformers langchain faiss-cpu pypdf tiktoken

# How It Works

The Pdfreader_openai operates in several stages:

1. It first processes a specified folder of PDF documents, extracting the text and splitting it into manageable chunks using Hugging Face Transformers library.
2. Each text chunk is then embedded using the default OpenAI embedding model (text-embedding-ada-002) through the LangChain library.
3. These embeddings are stored in a FAISS index, providing a compact and efficient storage method.
4. Finally, a query interface allows you to retrieve relevant information from the indexed data by asking questions. The application fetches and displays the most relevant text chunk.

