from google.colab import files #to read cv file in google colab
import os
import docx2txt
import pdfplumber
from langchain_openai import ChatOpenAI  

# Upload file
uploaded = files.upload()
file_path = list(uploaded.keys())[0]  # Get uploaded filename

# Ensure the file is properly saved in Colab’s environment
if not os.path.exists(file_path):
    print("Error: File not uploaded. Please try again.")
else:
    print(f"File '{file_path}' uploaded successfully!")

# Configure LangChain OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA"  
)

def extract_text_from_cv(file_path):
    """Extracts text from a given CV file (PDF, DOCX, or TXT)."""
    text = ""
    if file_path.endswith(".pdf"):
        with pdfplumber.open(file_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
    return text.strip()

def extract_info_from_cv(cv_text):
    """Uses LangChain's OpenAI model to extract structured CV details."""
    prompt = (f"Extract the following details from the given CV text:\n\n"
              f"- Full Name\n"
              f"- Phone Number\n"
              f"- Email Address\n"
              f"- Education\n"
              f"- Work Experience\n"
              f"- Skills\n"
              f"- Certificates\n\n"
              f"CV Content:\n{cv_text}\n\n")

    messages = [{"role": "user", "content": prompt}]
    response = llm.invoke(messages)

    return response.content  # Extracted CV data

#  Extract text after ensuring file upload
cv_text = extract_text_from_cv(file_path)

if cv_text:
    extracted_info = extract_info_from_cv(cv_text)
    print("\nExtracted CV Information:\n")
    print(extracted_info)
else:
    print("Error: Could not extract text from the CV. Please check the file format.")
