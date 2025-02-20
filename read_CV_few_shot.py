import json
from google.colab import files
import os
import docx2txt
import pdfplumber
from langchain_openai import ChatOpenAI  

#  Upload file
uploaded = files.upload()  # Prompts file upload
file_path = list(uploaded.keys())[0]  # Get uploaded filename

#  Ensure the file is properly saved in Colab’s environment
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
    """Uses LangChain's OpenAI model with Few-Shot Learning to extract structured CV details in JSON format."""
    
    few_shot_examples = [
        {
            "role": "system",
            "content": "You are an expert in extracting structured data from resumes. Format the output as JSON."
        },
        {
            "role": "user",
            "content": "Extract details from this CV:\n\nJohn Smith\nPhone: +1 234 567 890\nEmail: johnsmith@email.com\nEducation: B.Sc. Computer Science, MIT, 2019\nExperience: Software Engineer at Google (2020 - Present)\nSkills: Python, AI, Cloud Computing\nCertificates: AWS Certified Solutions Architect"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "Full Name": "John Smith",
                "Phone Number": "+1 234 567 890",
                "Email Address": "johnsmith@email.com",
                "Education": ["B.Sc. Computer Science, MIT, 2019"],
                "Work Experience": ["Software Engineer at Google (2020 - Present)"],
                "Skills": ["Python", "AI", "Cloud Computing"],
                "Certificates": ["AWS Certified Solutions Architect"]
            }, indent=4, ensure_ascii=False)
        },
        {
            "role": "user",
            "content": "Extract details from this CV:\n\nSarah Johnson\nPhone: +44 7890 123456\nEmail: sarahj@email.com\nEducation: MBA, Harvard Business School, 2018\nExperience: Product Manager at Amazon (2019 - Present)\nSkills: Leadership, Marketing, Strategy\nCertificates: PMP, Google Analytics Certification"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "Full Name": "Sarah Johnson",
                "Phone Number": "+44 7890 123456",
                "Email Address": "sarahj@email.com",
                "Education": ["MBA, Harvard Business School, 2018"],
                "Work Experience": ["Product Manager at Amazon (2019 - Present)"],
                "Skills": ["Leadership", "Marketing", "Strategy"],
                "Certificates": ["PMP", "Google Analytics Certification"]
            }, indent=4, ensure_ascii=False)
        }
    ]

    # provide the actual CV input
    actual_prompt = {
        "role": "user",
        "content": f"Extract details from this CV and return JSON format:\n\n{cv_text}"
    }

    # Combine few-shot examples with actual request
    messages = few_shot_examples + [actual_prompt]

    response = llm.invoke(messages)

    return response.content  # Extracted CV data

# Extract text after ensuring file upload
cv_text = extract_text_from_cv(file_path)

if cv_text:
    extracted_info = extract_info_from_cv(cv_text)

    try:
        # Convert the response to JSON format
        extracted_json = json.loads(extracted_info)
        print("\nExtracted CV Information (JSON Format):\n")
        print(json.dumps(extracted_json, indent=4, ensure_ascii=False))

    except json.JSONDecodeError:
        print("Error: Could not convert response to JSON. The model output may not be formatted correctly.")
else:
    print("Error: Could not extract text from the CV. Please check the file format.")
