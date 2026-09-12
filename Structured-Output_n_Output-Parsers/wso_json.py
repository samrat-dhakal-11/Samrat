from langchain_groq import ChatGroq
from pathlib import Path
from dotenv import load_dotenv

import os
# Load environment variables from .env file
path = Path(__file__).parent.parent.parent.parent / ".env"
load_dotenv(path)
api_key = os.getenv("GROQ_API_KEY")

modell=ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-20b",
    temperature=0
)

json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "processor": {
      "type": "string",
      "description": "The processor used in the device"
    },
    "work": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "List of tasks the device can perform well"
    },
    "device": {
      "type": "string",
      "description": "The name of the device being reviewed"
    },
    "battery": {
      "type": "integer",
      "description": "The battery capacity in mAh"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "review_by": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

Structured_model=modell.with_structured_output(json_schema)

result = Structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Samrat Dhakal
""")


print(result)
print(type(result))
print("Summary:",result['summary'])
print("Processor:",result['processor'])
print("Work:",result['work'])
print("Battery:",result['battery'])
print("Device:",result['device'])
print("Sentiment:",result['sentiment'])
print("Key Themes:",result['key_themes'])
print("Pros:",result['pros'])
print("Cons:",result['cons'])
print("Review by:",result['review_by'])

