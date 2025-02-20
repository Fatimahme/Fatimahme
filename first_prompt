from langchain_openai import ChatOpenAI  # pip install -U langchain_openai

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello world!"},
# ]
# # استفاده از مدل gpt-4o-mini
# model_name = "gpt-4o-mini"


# llm = ChatOpenAI(
#     model=model_name, base_url="https://api.avalai.ir/v1", api_key="aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA"
# )

# llm.invoke(messages)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA"  # Replace with your actual API key
)

# Get topic input from the user
topic = input("Enter a topic for the article: ")

# Create the prompt for OpenAI
messages = [
    {"role": "user", "content": f"Write an article on the topic: '{topic}'. "}
]

# Get the AI-generated article
response = llm.invoke(messages)

# Print only the article output
print("\nGenerated Article:\n")
print(response.content)
