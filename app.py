import os
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks

# 1. Initialize WorkspaceClient (handles auth natively in Databricks Apps)
w = WorkspaceClient()

# Get the index name injected from app.yml env variables
INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX") 

# Define a function to retrieve contexts using the WorkspaceClient
def retrieve_context(query: str, num_results: int = 3):
    if not INDEX_NAME:
        raise ValueError("VECTOR_SEARCH_INDEX environment variable is not set. Ensure the resource is linked in the UI and defined in app.yml.")
    
    # Query index directly through the SDK
    results = w.vector_search_indexes.query_index(
        index_name=INDEX_NAME,
        query_text=query,
        num_results=num_results,
        columns=["text"]  # Retrieve only the text column
    )
    
    # Extract page contents from the results
    contexts = []
    if results.result and results.result.data_array:
        for row in results.result.data_array:
            # Depending on how your index was built, row[0] or row[1] contains the text.
            # Usually the returned columns match the order in results.manifest.columns.
            contexts.append(row[1]) 
            
    return "\n".join(contexts)

# 2. Setup the LLM
llm = ChatDatabricks(endpoint="databricks-meta-llama-3-1-70b-instruct")

# 3. Streamlit UI
st.title("My RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Retrieve context using WorkspaceClient
        context = retrieve_context(prompt)
        
        full_prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {prompt}"
        
        response = llm.invoke(full_prompt)

        with st.chat_message("assistant"):
            st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        
    except Exception as e:
        st.error(f"Error executing chat sequence: {e}")
