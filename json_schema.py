from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# Schema

Json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": { "type": "string" },
            "description": "The key themes or topics mentioned in the review"
        },
        "review_text": {
            "type": "string",
            "description": "The text of the review"
        },
        "rating": {
            "type": "integer",
            "description": "The rating given by the user (1-5)"
        },
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"],
            "description": "The sentiment of the review"
        },
        "pros": {
            "type": "array",
            "items": { "type": "string" },
            "description": "A list of positive aspects mentioned in the review"
        },
        "cons": {
            "type": "array",
            "items": { "type": "string" },
            "description": "A list of negative aspects mentioned in the review"
        }
    },
    "required": ["review_text", "rating", "sentiment"]
}


structured_model = model.with_structured_output(Json_schema)

result = structured_model.invoke('''I bought the Sony WH-CH520 Wireless Headphones for $49.99, and overall, I am quite satisfied with my purchase. The headphones have a comfortable design and are lightweight enough to wear for several hours. The sound quality is clear, with good volume and decent bass for everyday listening. The battery life is also impressive, and the wireless connection works reliably without frequent interruptions.

### Pros

* Good sound quality for the price
* Lightweight and comfortable design
* Long battery life
* Easy Bluetooth connectivity
* Suitable for music, videos, and online meetings
* Attractive and simple design

### Cons

* The ear cushions could be more comfortable
* Bass may not be strong enough for users who prefer heavy bass
* The headphones do not have advanced noise cancellation
* The plastic build feels slightly less premium

Overall, the Sony WH-CH520 offers good value for its price. It is a suitable choice for everyday use, especially for people looking for affordable wireless headphones with good battery life and reliable sound quality.
''')

print(result)
