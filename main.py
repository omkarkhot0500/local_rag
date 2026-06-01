# =====================================================================
# RAG STEP 2: QUERYING AND GENERATION (main.py)
# =====================================================================
# This file handles the interactive loop where the user asks questions.
# 1. It imports the 'retriever' we defined in vector.py.
# 2. It queries that retriever using the user's question to pull relevant reviews.
# 3. It injects the reviews into a Prompt Template.
# 4. It passes the prompt to a local LLM (llama3.2) to generate a grounded answer.

# 1. IMPORTS
# OllamaLLM allows us to load and chat with large language models running locally on our computer.
from langchain_ollama.llms import OllamaLLM
# ChatPromptTemplate helps construct standard prompts, ensuring instructions and dynamic inputs
# (like reviews and questions) are formatted correctly for the LLM.
from langchain_core.prompts import ChatPromptTemplate
# We import the retriever from vector.py so we can perform similarity search on our Chroma database.
from vector import retriever

# 2. INITIALIZE THE LLM
# Here, we initialize Llama 3.2 (3 billion parameters), which runs completely locally via Ollama.
model = OllamaLLM(model="llama3.2")

# 3. DEFINE THE PROMPT TEMPLATE
# This is a template with placeholders ({reviews} and {question}).
# We instruct the AI about its persona and rules. This acts as the "system instructions".
# By inserting the retrieved reviews here, we prevent the model from hallucinating (making up answers).
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 4. CONSTRUCT THE CHAIN
# We use LangChain Expression Language (LCEL) to link components together.
# The pipe operator (|) takes the output of the first component (the formatted prompt)
# and feeds it directly as the input to the next component (the LLM model).
chain = prompt | model

# 5. USER INTERACTION LOOP
# We run an infinite loop to let the user ask multiple questions until they type 'q'.
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    
    # If the user enters 'q', we terminate the loop.
    if question == "q":
        break
    
    # --- RAG IN ACTION ---
    
    # A. RETRIEVAL PHASE:
    # We call retriever.invoke(question). The retriever converts the user's question to a vector,
    # compares it with our stored reviews in Chroma, and returns the 5 reviews closest in meaning.
    reviews = retriever.invoke(question)
    
    # B. GENERATION (AUGMENTATION) PHASE:
    # We execute the chain by feeding it:
    # 1. The 5 retrieved reviews we just fetched.
    # 2. The user's original question.
    # The chain automatically formats the prompt template and sends it to Llama 3.2.
    result = chain.invoke({"reviews": reviews, "question": question})
    
    # C. OUTPUT:
    # Print the model's response.
    print(result)