from langchain_openai import ChatOpenAI  
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA" 
)

# Get topic input from the user for examlpe music or astronomy or llm or blah blah blah :)))
topic = input("Enter a topic for the article: ")

# Create the prompt for OpenAI
messages = [
    {"role": "user", "content": f"Write an article on the topic: '{topic}'. "}
]

# Get the AI-generated article
response = llm.invoke(messages)
print("\nGenerated Article:\n")
print(response.content)
