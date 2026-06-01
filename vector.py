# =====================================================================
# RAG STEP 1: LOADING DATA, EMBEDDINGS, AND VECTOR STORAGE (vector.py)
# =====================================================================
# This file handles the "Indexing" phase of our Retrieval-Augmented Generation (RAG) system.
# Its job is to read our review database (CSV), convert the text reviews into mathematical 
# vectors (embeddings) that capture their semantic meaning, and store them in a Vector Database
# (Chroma) on disk. Finally, it creates a "Retriever" to fetch relevant reviews when asked questions.

# 1. IMPORTS
# langchain_ollama allows us to use Ollama models locally to turn text into embeddings.
from langchain_ollama import OllamaEmbeddings
# langchain_chroma is our lightweight vector database that runs locally and stores our embeddings.
from langchain_chroma import Chroma
# Document is LangChain's standard data structure. It wraps text and metadata together.
from langchain_core.documents import Document
import os
# pandas is used to load and parse structured CSV files.
import pandas as pd

# 2. LOAD DATA
# We load our restaurant reviews from the CSV file. pandas parses this into a DataFrame (table).
df = pd.read_csv("realistic_restaurant_reviews.csv")

# 3. INITIALIZE EMBEDDING MODEL
# An embedding model takes raw text and converts it into a long list of numbers (a vector).
# Text with similar semantic meaning (e.g., "The pizza was delicious" and "The crust was fantastic")
# will have vectors that are numerically close to each other.
# We are using "mxbai-embed-large" hosted locally on Ollama, which is optimized for search tasks.
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# 4. PREPARE THE DATABASE LOCATION
# We define a folder where Chroma will save the embedded reviews to disk.
db_location = "./chrome_langchain_db"

# Embedding text takes time and computation. To avoid re-embedding everything on every run,
# we check if the database folder already exists. If it does, we just load it instead of re-creating it.
add_documents = not os.path.exists(db_location)

# 5. PREPARE DOCUMENTS FOR THE VECTOR DATABASE
# If the database does not exist, we need to convert our raw CSV rows into LangChain Document objects.
if add_documents:
    documents = []
    ids = []  # Unique identifiers for each document
    
    # We iterate (loop) through each row of our reviews table
    for i, row in df.iterrows():
        # A Document has two main parts:
        # - page_content: The text that the vector database will search through.
        # - metadata: Extra info (like ratings or dates) that doesn't get searched, but can be filtered.
        document = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
# 6. INITIALIZE OR LOAD THE CHROMA VECTOR STORE
# We set up Chroma. 
# - collection_name is like a table name in a database.
# - persist_directory tells Chroma where to save the files.
# - embedding_function specifies which model to use to convert text into vectors.
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

# 7. ADD DOCUMENTS TO THE STORE
# If this is the first run (the database folder didn't exist), we now insert the documents.
# Chroma automatically takes the page_content of each document, passes it to the Ollama embedding
# function, obtains the numeric vectors, and saves them to the disk.
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)
    
# 8. CREATE THE RETRIEVER
# A "Retriever" is a lightweight interface wrapped around our vector store.
# When given a query (e.g., "Do they have gluten-free pizza?"), the retriever will:
# 1. Embed the query using the same embedding model.
# 2. Look up the vector database.
# 3. Retrieve and return the top 'k' most similar documents (in this case, the top 5 reviews).
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)