import openai
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the client
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    # List all available models (legacy method)
    models = openai.Model.list()
    
    print("✅ Your API key has access to these models:")
    for model in models["data"]:
        if "gpt" in model["id"] or "text" in model["id"]:  # Filter relevant models
            print(f"- {model['id']}")
    with open("models.txt", 'w') as file:
        for model in models["data"]:
            if "gpt" in model["id"] or "text" in model["id"]: 
                file.write(f"- {model['id']}\n")
    print(f"Files successfully written to {file}")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Check your API key or billing plan.")