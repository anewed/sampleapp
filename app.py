import streamlit as st
from databricks_langchain import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from databricks.ai_search.client import AISearchClient

# 1. Setup Retrieval
vsc = AISearchClient()
vector_search_index = vsc.get_index(
    endpoint_name="my_vector_search_endpoint",
    index_name="gg_test.dev.my_text_index"
)

# Initialize the retriever
vector_store = DatabricksVectorSearch(
    index=vector_search_index,
    text_column="text"
)
retriever = vector_store.as_retriever()

# 2. Setup the LLM (Using Databricks Foundation Model API)
# You can use "databricks-meta-llama-3-1-70b-instruct" or others available in your workspace
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

    # Retrieval and Generation
    docs = retriever.invoke(prompt)
    context = "\n".join([d.page_content for d in docs])
    
    full_prompt = f"Use the following context to answer the question:\n{context}\n\nQuestion: {prompt}"
    
    response = llm.invoke(full_prompt)

    with st.chat_message("assistant"):
        st.markdown(response.content)
    st.session_state.messages.append({"role": "assistant", "content": response.content})
