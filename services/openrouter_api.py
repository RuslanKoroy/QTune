from openai import OpenAI
import os

openrouter_client = None
OPENROUTER_KEY = ""

def update_openrouter_client(api_key=None):
    """Update the OpenRouter client with a new API key"""
    global openrouter_client, OPENROUTER_KEY
    if api_key:
        OPENROUTER_KEY = api_key
    if OPENROUTER_KEY:
        openrouter_client = OpenAI(
            api_key=OPENROUTER_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        openrouter_client = None

def generate(messages: list[dict], prompt_name: str, replace_dict: dict = None, model: str = None) -> str:
    if model is None:
        return "Error: No model specified for generation"
    
    prompt_file_path = f'{prompt_name}.md'
    try:
        with open(prompt_file_path, 'r', encoding='utf8') as f:
            system_message = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file '{prompt_file_path}' not found.")
        system_message = "You are a useful assistant."
    except Exception as e:
        print(f"Error reading prompt file '{prompt_file_path}': {e}")
        system_message = "You are a useful assistant."

    if replace_dict:
        for key, value in replace_dict.items():
            if value is not None:
                system_message = system_message.replace(key, str(value))
    
    full_messages = [{'role': 'system', 'content': system_message}] + list(messages)

    if openrouter_client is None:
        print("OpenRouter API key not set. Please set OPENROUTER_KEY in Settings.")
        return "API key not set. Please configure your OpenRouter API key in Settings."
    
    try:
        chat_completion = openrouter_client.chat.completions.create(
            model=model,
            messages=full_messages
        )
        generated_text = chat_completion.choices[0].message.content
        return generated_text

    except Exception as exception:
        print(f'Error in openrouter_generate with model {model}: {exception}')
        return f"Error generating response: {exception}"
