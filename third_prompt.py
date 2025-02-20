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
third_prompt = (f"Write an article on the topic: '{topic}', including the following sections: "
                 f"1. Introduction including introduce the '{topic}' and its importance in today world\n"
                 "2. Explanation of different methods like classic or the latest methods\n"
                 "3. Benefits and Challenges of every method with explanation\n"
                 "5. Conclusion of article and give prediction of the future of this field\n"
                 "Make it clear, well-structured, and engaging.")

# Choose which prompt to use
messages = [{"role": "user", "content": third_prompt}]  # Using the second prompt for better structure

# Get the AI-generated article
response = llm.invoke(messages)

# Print only the article output
print("\nGenerated Article:\n")
print(response.content)
