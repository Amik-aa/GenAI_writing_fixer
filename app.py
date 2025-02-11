import streamlit as st
import faiss
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Loading grammar correction model (BART)
MODEL_NAME = "facebook/bart-large-cnn"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Loading sentence embedding model for FAISS
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Predefined corrections for FAISS retrieval
corrections = [
    ("I has a apple.", "I have an apple."),
    ("She go to school.", "She goes to school."),
    ("He are very kind.", "He is very kind."),
]

# Creating FAISS index
correction_texts = [c[0] for c in corrections]
correction_embeddings = np.array([embed_model.encode(c[0]) for c in corrections]).astype('float32')
index = faiss.IndexFlatL2(correction_embeddings.shape[1])
index.add(correction_embeddings)

def retrieve_correction(text):
    """Retrieve similar corrections from FAISS"""
    embedding = np.array([embed_model.encode(text)]).astype('float32')
    _, idx = index.search(embedding, 1)
    return corrections[idx[0][0]][1]

def correct_grammar(text):
    """Fix grammatical errors using BART"""
    inputs = tokenizer("grammar: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def rewrite_text(text, style="formal"):
    """Rewrite text in formal or casual tone"""
    prompt = f"{style}: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI
st.title("✍️ AI-Powered Grammar & Writing Assistant")
st.write("Fix grammar mistakes, improve clarity, and change writing styles.")

# User input
text_input = st.text_area("Enter your text:", "This sentence are incorrect.")

# Style selection
style_option = st.radio("Choose style:", ["Grammar Correction", "Formal", "Casual"])

if st.button("Fix Text"):
    # First check FAISS for a quick correction
    faiss_correction = retrieve_correction(text_input)
    
    if style_option == "Grammar Correction":
        result = faiss_correction if faiss_correction else correct_grammar(text_input)
    else:
        result = rewrite_text(text_input, style=style_option.lower())
    
    st.subheader("🔹 Corrected Text:")
    st.write(result)
