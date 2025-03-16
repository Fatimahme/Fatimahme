from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.retrievers import ParentDocumentRetriever
import numpy as np
import os

# TF-IDF Retriever
class TfidfRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform([doc.page_content for doc in documents])

    def retrieve(self, query, top_k=5):
        query_vec = self.vectorizer.transform([query])
        scores = np.dot(self.tfidf_matrix, query_vec.T).toarray()
        ranked_indices = np.argsort(scores, axis=0)[::-1][:top_k]
        results = [self.documents[idx] for idx in ranked_indices.flatten()]
        return results

# Combined Retriever
class HybridRetriever:
    def __init__(self, tfidf_retriever, embedding_retriever):
        self.tfidf_retriever = tfidf_retriever
        self.embedding_retriever = embedding_retriever

    def retrieve(self, query, top_k=5):
        # Retrieve using both TF-IDF and Embedding models
        tfidf_results = self.tfidf_retriever.retrieve(query, top_k)
        embedding_results = self.embedding_retriever.get_relevant_documents(query)
        
        # Merge results (you can use different ranking strategies here)
        combined_results = list(set(tfidf_results + embedding_results))
        return combined_results

# Function to Initialize Retrievers
def initialize_retrievers(txt_documents, retriever_path):
    # Initialize Embedding-based Retriever
    embeddings = HuggingFaceEmbeddings()
    if os.path.exists(retriever_path):
        vectorstore = FAISS.load_local(retriever_path, embeddings)
    else:
        vectorstore = FAISS.from_documents(txt_documents, embeddings)
        vectorstore.save_local(retriever_path)

    embedding_retriever = ParentDocumentRetriever(vectorstore=vectorstore)

    # Initialize TF-IDF Retriever
    tfidf_retriever = TfidfRetriever(txt_documents)

    # Initialize Hybrid Retriever
    hybrid_retriever = HybridRetriever(tfidf_retriever, embedding_retriever)
    return hybrid_retriever

def main():
    txt_directory = "/content/daneshkar/"  # مسیر پوشه PDFهای شما
    metadata_path = "processed_files_metadata.pkl"
    retriever_path = "retriever_index"

    print("Processing new PDF files...")
    txt_documents = process_txt_files(txt_directory, metadata_path)

    if txt_documents:
        print(f"Processed {len(txt_documents)} chunks from new documents.")
    
    print("Initializing Hybrid Retriever...")
    hybrid_retriever = initialize_retrievers(txt_documents, retriever_path)
    print("Hybrid Retriever is ready to handle queries!")

    # نمونه جستجو
    query = "توضیحات مرتبط با موضوع مورد نظر"
    results = hybrid_retriever.retrieve(query, top_k=5)
    for result in results:
        print(f"Content: {result.page_content}\nMetadata: {result.metadata}\n")
