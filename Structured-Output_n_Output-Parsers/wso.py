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

n_result = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Samrat Dhakal

Write a structured output in JSON format with the following fields:
- key_themes: A list of key themes discussed in the review.
- summary: A brief summary of the review.
- device: The name of the device being reviewed.
- processor: The processor used in the device.
- sentiment: The sentiment of the review, either "pos" for positive, "neg"  for negative, or "neutral" for neutral.
- work: A list of tasks the device can perform well.
- pros: A list of pros of the device.
- cons: A list of cons of the device.
- battery: The battery capacity in mAh.
- review_by: The name of the reviewer.


Write a structured output in Pydantic format with the following fields:
- key_themes: A list of key themes discussed in the review.
- summary: A brief summary of the review.
- device: The name of the device being reviewed.
- processor: The processor used in the device.
- sentiment: The sentiment of the review, either "pos" for positive, "neg"  for negative, or "neutral" for neutral.
- work: A list of tasks the device can perform well.
- pros: A list of pros of the device.
- cons: A list of cons of the device.
- battery: The battery capacity in mAh.
- review_by: The name of the reviewer.

Write a structured output in Typedict format with the following fields:
- key_themes: A list of key themes discussed in the review.
- summary: A brief summary of the review.
- device: The name of the device being reviewed.
- processor: The processor used in the device.
- sentiment: The sentiment of the review, either "pos" for positive, "neg"  for negative, or "neutral" for neutral.
- work: A list of tasks the device can perform well.
- pros: A list of pros of the device.
- cons: A list of cons of the device.
- battery: The battery capacity in mAh.
- review_by: The name of the reviewer.
"""

print((modell.invoke(n_result)).content)



n_n_result = """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Samrat Dhakal
"""
print((modell.invoke(n_n_result)).content)