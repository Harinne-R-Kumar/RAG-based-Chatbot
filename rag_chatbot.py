import os
from typing import List, Dict, Any
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredImageLoader
)
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import magic
from PIL import Image
import cv2
import easyocr
import PyPDF2
import json
import pandas as pd
import numpy as np
import sys

# Load environment variables
load_dotenv()

# Configure Google API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

# Initialize vector store
persist_directory = "chroma_db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Initialize EasyOCR reader (do this once at startup)
reader = easyocr.Reader(['en'])  # Initialize for English. Add more languages if needed

def check_poppler_installation():
    """Check if poppler is installed and accessible."""
    try:
        # Try to convert a small test PDF
        test_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
        test_pdf.close()
        convert_from_path(test_pdf.name, first_page=1, last_page=1)
        os.unlink(test_pdf.name)
        return True
    except Exception as e:
        if "poppler" in str(e).lower():
            st.error("""
            Poppler is not installed or not in PATH. Please follow these steps:
            1. Download poppler from: https://github.com/oschwartz10612/poppler-windows/releases/
            2. Extract it to a location (e.g., C:\\Program Files\\poppler)
            3. Add the bin directory to your system's PATH environment variable
            4. Restart your application
            """)
            return False
        return True

def process_pdf_file(file_path: str) -> List[str]:
    """Process PDF file and extract text content using PyPDF2."""
    try:
        # Open the PDF file
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extract text from all pages
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Split the extracted text into chunks
            return text_splitter.split_text(text)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return []

def process_image_file(image_path: str) -> List[str]:
    """Process image file and extract text using EasyOCR."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'])  # Initialize for English
    except ImportError:
        st.error("EasyOCR is not installed. Installing it using: pip install easyocr")
        return []

    try:
        # Read image using PIL first to handle different formats
        image = Image.open(image_path)
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text using EasyOCR
        results = reader.readtext(np.array(image))
        
        # Get text from results with confidence filtering
        texts = [text for _, text, conf in results if conf > 0.5]
        
        if not texts:
            st.warning("No text detected in the image with sufficient confidence.")
            return []
            
        return text_splitter.split_text("\n".join(texts))
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return []

def process_video_file(file_path: str) -> List[str]:
    """Process video file and extract text from frames using EasyOCR."""
    try:
        import easyocr
        reader = easyocr.Reader(['en'])  # Initialize for English
    except ImportError:
        st.error("EasyOCR is not installed. Installing it using: pip install easyocr")
        return []

    cap = cv2.VideoCapture(file_path)
    text_chunks = []
    frame_count = 0
    max_frames = 100  # Limit the number of frames to process
    
    try:
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame to avoid redundancy
            if frame_count % 5 == 0:
                # Convert frame to RGB (EasyOCR expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Extract text using EasyOCR
                results = reader.readtext(frame_rgb)
                
                # Get text from results
                frame_texts = [text for _, text, conf in results if conf > 0.5]  # Filter by confidence
                if frame_texts:
                    text_chunks.extend(text_splitter.split_text("\n".join(frame_texts)))
            
            frame_count += 1
        
        return text_chunks
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return []
    finally:
        cap.release()

def process_file(file) -> List[str]:
    """Process different file types and extract text content."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        # Detect file type
        file_type = magic.from_file(tmp_file_path, mime=True)
        st.write(f"Processing file of type: {file_type}")
        
        if file_type == "application/pdf":
            st.info("Processing PDF file...")
            return process_pdf_file(tmp_file_path)
            
        elif file_type.startswith("image/"):
            st.info("Processing image file...")
            return process_image_file(tmp_file_path)
            
        elif file_type == "text/csv":
            st.info("Processing CSV file...")
            try:
                df = pd.read_csv(tmp_file_path)
                return text_splitter.split_text(df.to_string())
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
                return []
            
        elif file_type == "application/json":
            st.info("Processing JSON file...")
            try:
                with open(tmp_file_path, 'r') as f:
                    data = json.load(f)
                return text_splitter.split_text(json.dumps(data))
            except Exception as e:
                st.error(f"Error processing JSON: {str(e)}")
                return []
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            st.info("Processing DOCX file...")
            try:
                loader = UnstructuredWordDocumentLoader(tmp_file_path)
                documents = loader.load()
                return text_splitter.split_documents(documents)
            except Exception as e:
                st.error(f"Error processing DOCX: {str(e)}")
                st.error("Make sure python-docx and unstructured packages are installed.")
                return []
            
        elif file_type.startswith("video/"):
            st.info("Processing video file...")
            try:
                return process_video_file(tmp_file_path)
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
                st.error("Make sure OpenCV (cv2) is properly installed.")
                return []
            
        else:
            st.warning(f"Attempting to process unknown file type: {file_type}")
            try:
                with open(tmp_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                return text_splitter.split_text(text)
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return []
            
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

def initialize_vector_store():
    """Initialize or update the vector store with new documents."""
    return vectordb

def get_llm():
    """Initialize the language model."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

def get_qa_chain():
    """Create a question-answering chain."""
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3})
    )

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

# File uploader
uploaded_files = st.file_uploader(
    "Upload Documents (PDF, DOCX, CSV, JSON, Images, Videos)",
    type=["pdf", "docx", "csv", "json", "jpg", "jpeg", "png", "mp4", "avi"],
    accept_multiple_files=True
)

# Process uploaded files
if uploaded_files:
    with st.spinner("Processing documents..."):
        for file in uploaded_files:
            chunks = process_file(file)
            # Only add non-empty chunks to the vector database
            if chunks:  # Check if chunks list is not empty
                try:
                    vectordb.add_texts(chunks)
                    st.success(f"Successfully processed {file.name}")
                except Exception as e:
                    st.error(f"Error adding {file.name} to database: {str(e)}")
            else:
                st.warning(f"No text content extracted from {file.name}")
        st.success("Document processing completed!")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            qa_chain = get_qa_chain()
            response = qa_chain.run(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response}) 