import json
import os
from transformers import AutoTokenizer

def fetch_template_from_model(model_name):
    try:
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get the chat template if it exists
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            template = tokenizer.chat_template
            return True, template
        else:
            # Return a default template if none exists
            default_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% elif message['role'] == 'system' %}
System: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
Assistant:
{% endif %}"""
            return True, default_template
            
    except Exception as e:
        # Return a default template if we can't fetch from the model
        default_template = """{% for message in messages %}
{% if message['role'] == 'user' %}
User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}
Assistant: {{ message['content'] }}
{% elif message['role'] == 'system' %}
System: {{ message['content'] }}
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
Assistant:
{% endif %}"""
        return False, f"Could not fetch template from model: {str(e)}. Using default template."

def save_custom_template(template_content, template_name="custom_template"):
    try:
        # Create templates directory if it doesn't exist
        templates_dir = "templates"
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
        
        # Save the template
        template_path = os.path.join(templates_dir, f"{template_name}.txt")
        with open(template_path, "w", encoding="utf-8") as f:
            f.write(template_content)
        
        return True, f"Custom template saved successfully to {template_path}"
        
    except Exception as e:
        return False, f"Error saving template: {str(e)}"

def load_custom_template(template_name="custom_template"):
    try:
        template_path = os.path.join("templates", f"{template_name}.txt")
        
        if not os.path.exists(template_path):
            return False, f"Template file not found: {template_path}"
        
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()
        
        return True, template_content
        
    except Exception as e:
        return False, f"Error loading template: {str(e)}"

def list_available_templates():
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        return []
    
    templates = []
    for file in os.listdir(templates_dir):
        if file.endswith(".txt"):
            templates.append(file[:-4])  # Remove .txt extension
    
    return templates

def apply_template_to_dataset(dataset_path, template_name="custom_template"):
    try:
        # Load the dataset
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        # Load the template
        success, template = load_custom_template(template_name)
        if not success:
            return False, template  # template contains the error message
        
        # In a real implementation, we would apply the template to the dataset
        # For now, we'll just return a success message
        message = f"Template '{template_name}' would be applied to dataset '{dataset_path}'.\n"
        message += "In a full implementation, this would format the dataset according to the template."
        
        return True, message
        
    except Exception as e:
        return False, f"Error applying template to dataset: {str(e)}"
