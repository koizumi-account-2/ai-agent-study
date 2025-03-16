from dotenv import load_dotenv
from openai import OpenAI
import os
import json
load_dotenv()

api_key = os.getenv("API_KEY")

client = OpenAI(api_key=api_key)


def get_weather(location,unit="fahrenheit"):
    if "tokyo" in location.lower():
        return json.dumps({"weather": "晴れ", "temperature": 20,"unit": unit})
    elif "osaka" in location.lower():
        return json.dumps({"weather": "晴れ", "temperature": 22,"unit": unit})
    elif "sapporo" in location.lower():
        return json.dumps({"weather": "晴れ", "temperature": 18,"unit": unit})
    else:
        return json.dumps({"weather": "晴れ", "temperature": 25,"unit": unit})


available_functions = {
    "get_weather": get_weather
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The location to get the weather for"},
                    "unit": {"type": "string", "description": "The unit of temperature to return", "enum": ["fahrenheit", "celsius"]}
                },
                "required": ["location"]
            }
        }
    }
]

messages = [
    {"role": "user", "content": "今日の東京の天気を教えて"}
]
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

print(response.choices[0].message.tool_calls)
response_message = response.choices[0].message
messages.append(response_message.to_dict())

for tool_call in response_message.tool_calls:
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    function_result = available_functions[function_name](**function_args)
    messages.append({
        "tool_call_id": tool_call.id,
        "role": "tool",
        "name": function_name,
        "content": function_result
    }
)

print(json.dumps(messages, indent=2 , ensure_ascii=False))

second_response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print(second_response.choices[0].message.content)