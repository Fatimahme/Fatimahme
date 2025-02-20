from langchain_openai import ChatOpenAI  
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.avalai.ir/v1",
    api_key=  "aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA"
)

# my choice is to get the scientific news input from the user and change it into an advertisment
scientific_news = input("Enter a scientific news article: ")

# Zero-shot prompt to transform it into an engaging advertisement
prompt = (f"Transform the following scientific news into an engaging and persuasive advertisement. "
          f"Make it exciting, appealing to a wide audience, and highlight its benefits in a compelling way.\n\n"
          f"Scientific News:\n{scientific_news}\n\n"
          f"Advertisement:")

# Send the request to OpenAI
messages = [{"role": "user", "content": prompt}]
response = llm.invoke(messages)
print("\nGenerated Advertisement:\n")
print(response.content)
