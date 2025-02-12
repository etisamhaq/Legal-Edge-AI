import streamlit as st
from rag_system import RAGSystem  # Assuming you have the above RAG code in a file named rag_system.py

# Initialize the RAG system
rag = RAGSystem(
    neo4j_uri="bolt://localhost:7687", 
    neo4j_user="neo4j", 
    neo4j_password="12345678", 
    groq_api_key="your_api_key_here"
)

# Streamlit UI
st.title("Graph-Based RAG Chatbot")
st.write("Ask me anything related to legal articles!")

# Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_query := st.chat_input("Type your question here..."):
    st.session_state["messages"].append({"role": "user", "content": user_query})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate response
    with st.chat_message("assistant"):
        response = rag.process_query(user_query)
        st.session_state["messages"].append({"role": "assistant", "content": response})
        st.markdown(response)

# Cleanup
rag.close()
