from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import chain
import json  # Import the json module

# 1. Define the Output Schema
class SentimentScore(BaseModel):
    """Sentiment analysis result."""
    score: float = Field(description="Sentiment score between -1 (negative) and 1 (positive)")
    explanation: str = Field(description="Brief explanation of why this score was given")

# 2. Create a Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a sentiment analysis expert.  Analyze the sentiment of the provided text and provide a numerical score between -1 and 1, and a brief explanation.  Respond in JSON format.",
        ),
        (
            "user",
            "Analyze the sentiment of the following text:\n\n{text}",
        ),
    ]
)   

# 3. Instantiate the Llama 3.2 1B model
llm = OllamaLLM(model="llama3.2:1b", temperature=0.0) # Make sure you have this model downloaded

# 4. Create the output parser
parser = JsonOutputParser(pydantic_object=SentimentScore)

# 5. Create the chain
chain = prompt | llm | parser

# 6. Example Usage
text = "This is an amazing product! I love it."
result = chain.invoke({"text": text})

print(f"Sentiment Score: {result.score}")
print(f"Explanation: {result.explanation}")
