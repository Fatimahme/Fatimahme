from langchain_openai import ChatOpenAI  
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA" 
)

# Get topic input from the user for examlpe music or astronomy or llm or blah blah blah :)))
topic = input("Enter a topic for the article: ")


third_prompt = (f"Write an article on the topic: '{topic}', including the following sections: "
                 f"1. Introduction including introduce the '{topic}' and its importance in today world\n"
                 "2. Explanation of different methods like classic or the latest methods\n"
                 "3. Benefits and Challenges of every method with explanation\n"
                 "5. Conclusion of article and give prediction of the future of this field\n"
                 "Make it clear, well-structured, and engaging.")


messages = [{"role": "user", "content": third_prompt}]  

# Get the AI-generated article
response = llm.invoke(messages)
print("\nGenerated Article:\n")
print(response.content)
