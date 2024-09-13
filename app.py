import streamlit as st
import pdfplumber
from pinecone import Pinecone
import time
import cohere
import numpy as np
from pinecone import ServerlessSpec

# Initialize Pinecone and Cohere clients
api_key = "8195b7e2-f5a8-4f29-84b5-f0d43bdcd888"
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud='aws', region='us-east-1')
index_name = 'resume-retrieval-augmentation-fast'

cohere_client = cohere.Client('VoGQ1PX4QgbUh0v6ZfdK9QwNZVAIvBj7WUGHZv7a')

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def setup_index():
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)
    
    pc.create_index(
        index_name,
        dimension=4096,
        metric='dotproduct',
        spec=spec
    )
    
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    return pc.Index(index_name)

def embed_and_upsert_documents(documents, index):
    embeddings = cohere_client.embed(texts=documents).embeddings
    vectors = [{'id': str(i), 'values': emb} for i, emb in enumerate(embeddings)]
    index.upsert(vectors)

def retrieve_documents(query, index, top_k=5):
    query_embedding = cohere_client.embed(texts=[query]).embeddings[0]
    for i, val in enumerate(query_embedding):
        if not isinstance(val, float) or not np.isfinite(val):
            query_embedding[i] = 0.0

    query_embedding = [float(np.clip(val, -1, 1)) for val in query_embedding]
    result = index.query(vector=[query_embedding], top_k=top_k)
    return [documents[int(match['id'])] for match in result['matches']]

def generate_answer(query, context):
    combined_context = ' '.join(context)
    response = cohere_client.generate(
        model='command-r-plus-08-2024',
        prompt=f"Context: {combined_context}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens=100
    )
    return response.generations[0].text

# Streamlit app
st.title("Resume Chatbot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.write("Processing PDF...")
    resume_text = extract_text_from_pdf(uploaded_file)
    documents = resume_text.split('\n')

    index = setup_index()
    embed_and_upsert_documents(documents, index)
    
    st.write("PDF processed and data indexed. You can now ask questions about the resume.")

    query = st.text_input("Enter your question:")

    if query:
        context = retrieve_documents(query, index)
        answer = generate_answer(query, context)
        st.write(f"Answer: {answer}")
