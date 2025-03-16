!pip install PyPDF2
import os
import pickle
from datetime import datetime
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Step 1: Upload and process text (PDF) files
def process_txt_files(txt_directory, metadata_path, chunk_size=500, chunk_overlap=50):
    # Load list of already processed files
    if os.path.exists(metadata_path):
        with open(metadata_path, "rb") as f:
            processed_metadata = pickle.load(f)
    else:
        processed_metadata = {}

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    txt_documents = []
    for txt_file in os.listdir(txt_directory):
        if txt_file.endswith(".pdf") and txt_file not in processed_metadata:
            file_path = os.path.join(txt_directory, txt_file)
            reader = PdfReader(file_path)
            txt_content = ""
            for page in reader.pages:
                txt_content += page.extract_text()
            
            # Split text into smaller chunks
            txt_chunks = text_splitter.split_text(txt_content)
            
            # Add metadata (identifier ID, timestamp, file name)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            identifier_id = f"{txt_file}_{current_time}"
            documents = [
                Document(page_content=chunk, metadata={
                    "id": identifier_id,
                    "uploaded_at": current_time,
                    "source": txt_file
                })
                for chunk in txt_chunks
            ]
            txt_documents.extend(documents)
            
            # Mark file as processed
            processed_metadata[txt_file] = {
                "uploaded_at": current_time,
                "id": identifier_id
            }

    # Save metadata to file
    with open(metadata_path, "wb") as f:
        pickle.dump(processed_metadata, f)

    return txt_documents

# Updated Step: Use ParentDocumentRetriever with stored documents
def store_and_retrieve_with_parent_retriever(txt_documents, retriever_path):
    from langchain.vectorstores import FAISS
    from langchain.retrievers import ParentDocumentRetriever
    from langchain.embeddings import HuggingFaceEmbeddings
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings()
    
    # Save documents into FAISS index
    if os.path.exists(retriever_path):
        vectorstore = FAISS.load_local(retriever_path, embeddings)
    else:
        vectorstore = FAISS.from_documents(txt_documents, embeddings)
    
    vectorstore.save_local(retriever_path)
    
    # Wrap vectorstore with ParentDocumentRetriever
    retriever = ParentDocumentRetriever(vectorstore=vectorstore)
    return retriever


def main():
    txt_directory = "/content/daneshkar/"  
    metadata_path = "processed_files_metadata.pkl"
    retriever_path = "retriever_index"
    
    print("Processing new PDF files...")
    txt_documents = process_txt_files(txt_directory, metadata_path)

    if txt_documents:
        print(f"Processed {len(txt_documents)} chunks from new documents.")
        
        print("Storing documents and initializing ParentDocumentRetriever...")
        retriever = store_and_retrieve_with_parent_retriever(txt_documents, retriever_path)
        print("ParentDocumentRetriever is ready to handle queries!")
    else:
        print("No new files to process.")

if __name__ == "__main__":
    main()
