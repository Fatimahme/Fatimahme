from langchain_openai import ChatOpenAI  # Ensure langchain_openai is installed

# Configure the LLM with your API details
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA"  # Replace with your actual API key
)

# Get topic input from the user
topic = input("Enter a topic for the article: ")

# Define the second prompt for a more structured response
second_prompt = (f"Write an article on the topic: '{topic}', including the following sections: "
                 "1. Introduction\n"
                 "2. Explanation of different methods\n"
                 "3. Benefits\n"
                 "4. Challenges\n"
                 "5. Conclusion\n")

# Choose which prompt to use
messages = [{"role": "user", "content": second_prompt}]  # Using the second prompt for better structure

# Get the AI-generated article
response = llm.invoke(messages)

# Print only the article output
print("\nGenerated Article:\n")
print(response.content)
