from langchain_openai import ChatOpenAI  
llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-Z0DGD0GsA8Y59MHb92ZUKfSPwWRpBUrfObPOX8t83VeYX0vA" 
)

# Get the scientific news input from the user
scientific_news = input("Enter a scientific news article: ")

# Few-shot examples to guide the model to change it into an advertisment
examples = [
    {
        "role": "user",
        "content": "Scientists have discovered a new vaccine that boosts immunity against the flu by 60%."
    },
    {
        "role": "assistant",
        "content": "💉 **Stay Healthy, Stay Strong!** 🏥\nIntroducing the latest **breakthrough flu vaccine**, now with a **60% immunity boost**! ❄️ No more sick days—just pure protection and peace of mind. Get vaccinated today and power up your immune system! 🔬💪"
    },
    {
        "role": "user",
        "content": "A new electric car battery charges in just 10 minutes, making long road trips more convenient."
    },
    {
        "role": "assistant",
        "content": "⚡ **Charge in Minutes, Drive for Miles! 🚗**\nSay goodbye to long charging times and hello to **10-minute full charges**! With our revolutionary **fast-charging EV battery**, road trips just got easier. More power, less waiting—**the future of driving is here!** 🔋🌍"
    }
]

# Create a few-shot prompt with the user's input
messages = examples + [
    {"role": "user", "content": f"Transform the following scientific news into an engaging advertisement:\n{scientific_news}"}
]

# Get the AI-generated advertisement
response = llm.invoke(messages)
print("\nGenerated Advertisement:\n")
print(response.content)
