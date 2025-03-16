
import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# Step 1: Read and process PDF files
def process_pdfs(pdf_directory):
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append(text)
    return documents

# Step 2: Create vector database from processed content
def create_vector_store(documents, index_path, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index and documents
    faiss.write_index(index, index_path)
    with open("documents.pkl", "wb") as f:
        pickle.dump(documents, f)

# Step 3: Load vector store and use it
def load_vector_store(index_path):
    index = faiss.read_index(index_path)
    with open("documents.pkl", "rb") as f:
        documents = pickle.load(f)
    return index, documents

# Main program
def main():
    pdf_directory = "/content/"  # Change to your folder with PDFs
    index_path = "vector_store.index"
    
    if not os.path.exists(index_path):
        print("Processing PDFs and creating vector store...")
        documents = process_pdfs(pdf_directory)
        create_vector_store(documents, index_path)
    else:
        print("Loading existing vector store...")
        index, documents = load_vector_store(index_path)

    print("Vector store is ready for use!")

if __name__ == "__main__":
    main()
