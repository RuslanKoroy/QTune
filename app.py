import gradio as gr
import os
import json
import torch
from huggingface_hub import list_models, model_info
import psutil
import GPUtil
import requests

# Import our existing modules
from services.openrouter_api import generate

from services.create_dataset import create_dataset
from services.train_model import train_model, merge_lora_model
from services.model_converter import convert_to_gguf, push_to_ollama
from services.template_manager import fetch_template_from_model, save_custom_template

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"

# Global variables to store state
current_model = None
current_tokenizer = None
training_status = "Not started"

# Global variables for API keys
openrouter_key = ""

# File to store API keys
API_KEYS_FILE = "api_keys.json"

# Load API keys from file at startup
def load_api_keys():
    """Load API keys from file"""
    global openrouter_key
    try:
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, 'r') as f:
                keys = json.load(f)
                openrouter_key = keys.get("openrouter_key", "")
                # Update the OpenRouter client with the loaded key
                if openrouter_key:
                    from services.openrouter_api import update_openrouter_client
                    update_openrouter_client(openrouter_key)
    except Exception as e:
        print(f"Error loading API keys: {e}")

# Model selection history (last 5 models)
model_history = []

# Function to fetch models from Hugging Face
def fetch_hf_models():
    try:
        # Fetch popular language models
        models = list_models(filter="text-generation", sort="downloads", direction=-1, limit=20)
        model_names = [model.id for model in models]
        return model_names
    except Exception as e:
        print(f"Error fetching models from Hugging Face: {e}")
        # Return default models if API fails
        return [
            "google/gemma-3-4b-it",
            "google/gemma-3-2b-it"
        ]

# Function to fetch models from OpenRouter
def fetch_openrouter_models():
    try:
        url = "https://openrouter.ai/api/v1/models"
        headers = {}
        # Add authentication header if API key is available
        if openrouter_key:
            headers["Authorization"] = f"Bearer {openrouter_key}"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # Extract model IDs from the response
            model_names = [model["id"] for model in data["data"]]
            # Filter for text generation models and sort by popularity
            text_models = [model for model in model_names if "embedding" not in model.lower()]
            return text_models[:50]  # Return top 50 models
        else:
            print(f"Error fetching models from OpenRouter: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching models from OpenRouter: {e}")
        return []

# Function to search models
def search_models(query, all_models):
    """Filter models based on search query"""
    if not query:
        return all_models[:50]  # Return top 50 models if no query
    filtered_models = [model for model in all_models if query.lower() in model.lower()]
    return filtered_models[:50]  # Return top 50 matching models

# Function to search across all available models (HF and OpenRouter)
def search_all_models(query):
    """Search across all available models from both HuggingFace and OpenRouter"""
    all_models = []
    
    # Get HuggingFace models
    try:
        hf_models = fetch_hf_models()
        all_models.extend(hf_models)
    except Exception as e:
        print(f"Error fetching HuggingFace models: {e}")
    
    # Get OpenRouter models
    try:
        or_models = fetch_openrouter_models()
        all_models.extend(or_models)
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}")
    
    # Search through all models
    return search_models(query, all_models)

# Function to add model to history
def add_to_model_history(model_name):
    """Add model to history, keeping only the last 5"""
    global model_history
    if model_name in model_history:
        model_history.remove(model_name)
    model_history.insert(0, model_name)
    model_history = model_history[:5]  # Keep only last 5
    return model_history

# Function to validate if a model exists on HuggingFace
def validate_hf_model(model_name):
    """Check if a model exists on HuggingFace"""
    try:
        # Try to fetch model info
        from huggingface_hub import model_info as hf_model_info
        hf_model_info(model_name)
        return True
    except Exception as e:
        print(f"Model {model_name} not found on HuggingFace: {e}")
        return False

# Function to validate if a model exists on OpenRouter
def validate_or_model(model_name):
    """Check if a model exists on OpenRouter"""
    try:
        # Fetch OpenRouter models and check if the model is in the list
        or_models = fetch_openrouter_models()
        return model_name in or_models
    except Exception as e:
        print(f"Error checking model {model_name} on OpenRouter: {e}")
        return False

# Function to validate model existence on both platforms
def validate_model(model_name):
    """Check if a model exists on either HuggingFace or OpenRouter"""
    if not model_name or model_name == "":
        return "Please enter a model name", "red"
    
    # Check HuggingFace first
    if validate_hf_model(model_name):
        return f"✅ Model '{model_name}' found on HuggingFace", "green"
    
    # Check OpenRouter if not found on HuggingFace
    if validate_or_model(model_name):
        return f"✅ Model '{model_name}' found on OpenRouter", "green"
    
    # If not found on either platform
    return f"❌ Model '{model_name}' not found on HuggingFace", "red"

# Function to load model info
def load_model_info(model_name):
    info = f"Model: {model_name}\n"
    info += f"Device: {DEVICE}\n"
    if CUDA_AVAILABLE:
        info += f"CUDA Version: {torch.version.cuda}\n"
        info += f"GPU: {torch.cuda.get_device_name(0)}\n"
        info += f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n"
    else:
        info += "Running on CPU (training will be slow)\n"
    return info

# Function to create dataset from conversation builder
def create_dataset_gradio(conversation_history, system_prompt, model_name, output_file, num_examples, append_to_existing=False):
    # Save system prompt to temporary file
    with open("temp_system_prompt.md", "w", encoding="utf-8") as f:
        f.write(system_prompt)
    
    try:
        dataset = []
        
        # If appending to existing dataset, load existing data
        if append_to_existing and os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf8') as f:
                    dataset = json.load(f)
            except Exception as e:
                print(f"Error loading existing dataset: {e}")
                dataset = []  # Start fresh if there's an error loading existing data
        
        # Generate multiple examples based on conversation history
        for i in range(num_examples):
            # Convert conversation history to messages format
            messages = []
            for user_msg, assistant_msg in conversation_history:
                if user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
            
            # Generate response based on the full conversation history
            from services.openrouter_api import generate
            assistant_response = generate(
                messages=messages,
                prompt_name="temp_system_prompt",
                model=model_name
            )
            
            if assistant_response:
                # Create dataset entry with full conversation + generated response
                dataset_entry = {
                    "messages": [
                        {"role": "system", "content": system_prompt}
                    ]
                }
                
                # Add all messages from conversation history
                for user_msg, assistant_msg in conversation_history:
                    if user_msg:
                        dataset_entry["messages"].append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        dataset_entry["messages"].append({"role": "assistant", "content": assistant_msg})
                
                # Add the generated response as the final assistant message
                dataset_entry["messages"].append({"role": "assistant", "content": assistant_response})
                
                dataset.append(dataset_entry)
        
        # Save dataset to file
        with open(output_file, 'w', encoding='utf8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        # Read the created dataset
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                dataset_content = f.read()
            action = "appended to" if append_to_existing and os.path.exists(output_file) else "saved to"
            # Return status message and parsed dataset for preview
            parsed_dataset = preview_dataset_content(dataset_content)
            return f"Dataset {action} successfully! Generated {num_examples} examples. Total entries: {len(dataset)}. Saved to {output_file}", parsed_dataset
        else:
            return "Error: Dataset file was not created", ""
    except Exception as e:
        return f"Error creating dataset: {str(e)}", ""

# Function to load dataset
def load_dataset_preview(dataset_file):
    try:
        if dataset_file is None:
            return "Please upload a dataset file"
        
        if dataset_file.name.endswith('.json'):
            with open(dataset_file.name, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            # Prepare data for DataFrame
            data = []
            headers = ["Entry", "Role", "Content"]
            
            # Show first few entries
            for i, entry in enumerate(dataset[:5]):  # Show first 5 entries
                for message in entry['messages']:
                    data.append([f"Entry {i+1}", message['role'], message['content'][:200] + "..." if len(message['content']) > 200 else message['content']])
            
            return { "headers": headers, "data": data }
        else:
            return "Unsupported file format. Please upload a JSON file."
    except Exception as e:
        return f"Error loading dataset: {str(e)}"
# Function to preview dataset content (for use with string content)
def preview_dataset_content(dataset_content):
    try:
        if not dataset_content:
            return "No dataset content to preview"
        
        # Parse the JSON content
        dataset = json.loads(dataset_content)
        
        # Prepare data for DataFrame
        data = []
        headers = ["Entry", "Role", "Content"]
        
        # Show first few entries
        for i, entry in enumerate(dataset[:5]):  # Show first 5 entries
            for message in entry['messages']:
                data.append([f"Entry {i+1}", message['role'], message['content'][:200] + "..." if len(message['content']) > 200 else message['content']])
        
        return { "headers": headers, "data": data }
    except Exception as e:
        return f"Error previewing dataset: {str(e)}"

# Training functions
def start_training_wrapper(model_name, dataset_file, lora_r, lora_alpha, lora_dropout,
                         target_modules, num_epochs, batch_size, grad_accum, learning_rate,
                         gradient_checkpointing, fp16):
    global training_status
    
    # Handle dataset path
    if dataset_file is None:
        return "Please select a dataset file"
    
    dataset_file_path = dataset_file.name if hasattr(dataset_file, 'name') else dataset_file
    
    # Convert target_modules to proper format
    if target_modules == "all-linear":
        target_modules = "all-linear"
    elif target_modules == "q_proj,v_proj":
        target_modules = ["q_proj", "v_proj"]
    elif target_modules == "k_proj,o_proj":
        target_modules = ["k_proj", "o_proj"]
    
    # Start training
    success, message = train_model(
        model_id=model_name,
        dataset_path=dataset_file_path,
        output_dir="tuned_model",
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        gradient_checkpointing=gradient_checkpointing,
        fp16=fp16
    )
    
    training_status = message
    return message

def stop_training():
    return "Training stopped."

# Model conversion functions
def convert_to_gguf_wrapper(model_path, quantization_type, output_format):
    # Convert output_format to file extension
    ext = ".gguf" if output_format == "gguf" else ".bin"
    
    # Generate output path
    model_name = os.path.basename(model_path)
    output_path = f"converted_models/{model_name}{ext}"
    
    # Call the actual conversion function
    success, message = convert_to_gguf(model_path, output_path, quantization_type)
    return message

def push_to_ollama_wrapper(model_path, ollama_model_name):
    # Call the actual Ollama push function
    success, message = push_to_ollama(model_path, ollama_model_name)
    return message

# Template management functions
def fetch_template_from_model_wrapper(model_name):
    success, template = fetch_template_from_model(model_name)
    if success:
        return template
    else:
        return f"Error: {template}"

def save_custom_template_wrapper(template_content):
    success, message = save_custom_template(template_content, "custom_template")
    return message

# Conversation builder functions
def add_user_message(history, message):
    """Add user message to conversation history"""
    if history is None:
        history = []
    if message:
        history.append((message, None))
    return history, ""

def add_assistant_message(history, message):
    """Add assistant message to conversation history"""
    if history is None:
        history = []
    if message and len(history) > 0:
        # Update the last entry with the assistant message
        history[-1] = (history[-1][0], message)
    return history, ""

def clear_conversation():
    """Clear conversation history"""
    return [], "", ""

def save_conversation_template(history, template_name):
    """Save conversation as a template"""
    if not template_name:
        return "Please enter a template name"
    
    if not history:
        return "Conversation is empty"
    
    try:
        # Convert history to JSON format
        template_data = []
        for user_msg, assistant_msg in history:
            if user_msg:
                template_data.append({"role": "user", "content": user_msg})
            if assistant_msg:
                template_data.append({"role": "assistant", "content": assistant_msg})
        
        # Save to file
        templates_dir = "templates"
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
        
        template_path = os.path.join(templates_dir, f"{template_name}.json")
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(template_data, f, ensure_ascii=False, indent=2)
        
        return f"Template '{template_name}' saved successfully!"
    except Exception as e:
        return f"Error saving template: {str(e)}"

def load_conversation_templates():
    """Load available conversation templates"""
    templates_dir = "templates"
    if not os.path.exists(templates_dir):
        return []
    
    templates = []
    for file in os.listdir(templates_dir):
        if file.endswith(".json"):
            templates.append(file[:-5])  # Remove .json extension
    return templates

# Settings functions
def save_api_keys(openrouter_key_input):
    """Save API keys to environment variables and file"""
    global openrouter_key
    if openrouter_key_input:
        openrouter_key = openrouter_key_input
        os.environ["OPENROUTER_KEY"] = openrouter_key_input
        # Update the OpenRouter client with the new key
        from services.openrouter_api import update_openrouter_client
        update_openrouter_client(openrouter_key_input)
        # Save to file
        try:
            keys = {"openrouter_key": openrouter_key_input}
            with open(API_KEYS_FILE, 'w') as f:
                json.dump(keys, f)
            return "API keys saved successfully!"
        except Exception as e:
            return f"API keys saved to environment, but failed to save to file: {e}"
    return "Please enter a valid OpenRouter API key."

def validate_openrouter_key(key_to_test):
    """Validate the OpenRouter API key"""
    if not key_to_test:
        return "No API key provided"
    
    try:
        # Test the key by fetching models
        url = "https://openrouter.ai/api/v1/models"
        headers = {"Authorization": f"Bearer {key_to_test}"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            return "✅ API key is valid"
        elif response.status_code == 401:
            return "❌ API key is invalid"
        else:
            return f"⚠️ Unexpected response: {response.status_code}"
    except Exception as e:
        return f"❌ Error validating key: {str(e)}"

def get_system_info():
    """Get system information"""
    info = "## System Information\n\n"
    
    # CPU Info
    info += f"**CPU:** {psutil.cpu_count()} cores\n"
    info += f"**RAM:** {psutil.virtual_memory().total / (1024**3):.1f} GB\n"
    
    # GPU Info
    if CUDA_AVAILABLE:
        info += f"**GPU:** {torch.cuda.get_device_name(0)}\n"
        info += f"**VRAM:** {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\n"
        info += f"**CUDA Version:** {torch.version.cuda}\n"
    else:
        # Try to get GPU info using GPUtil
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info += f"**GPU:** {gpu.name}\n"
                info += f"**VRAM:** {gpu.memoryTotal / 1024:.1f} GB\n"
            else:
                info += "**GPU:** Not available (running on CPU)\n"
        except:
            info += "**GPU:** Not available (running on CPU)\n"
    
    # Disk Space
    disk_usage = psutil.disk_usage('/')
    info += f"**Disk Space:** {disk_usage.total / (1024**3):.1f} GB total\n"
    
    return info

def get_help_info():
    """Get help information"""
    try:
        with open("help_text.md", "r", encoding="utf-8") as f:
            help_text = f.read()
        return help_text
    except FileNotFoundError:
        return "Help file not found."
    except Exception as e:
        return f"Error reading help file: {str(e)}"

# Load API keys at startup
load_api_keys()

# Main Gradio interface
with gr.Blocks(title="QTune", css="""
    .main-action-btn {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        border: none;
        color: white;
        padding: 12px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .main-action-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .model-history-btn {
        margin: 2px;
        font-size: 12px;
        padding: 4px 8px;
    }
""") as demo:
    gr.Markdown("# 🚀 QTune")
    gr.Markdown("### Fine-tune language models on consumer GPU")
    
    with gr.Tabs():
        # Model Selection Tab
        with gr.TabItem("🤖 Model Selection"):
            gr.Markdown("## Select a Model for Fine-tuning")
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_history_container = gr.Column(visible=False)
                    with model_history_container:
                        gr.Markdown("### 🕒 Recent Models")
                        model_history_btn1 = gr.Button(visible=False, variant="secondary", size="sm", elem_classes=["model-history-btn"])
                        model_history_btn2 = gr.Button(visible=False, variant="secondary", size="sm", elem_classes=["model-history-btn"])
                        model_history_btn3 = gr.Button(visible=False, variant="secondary", size="sm", elem_classes=["model-history-btn"])
                        model_history_btn4 = gr.Button(visible=False, variant="secondary", size="sm", elem_classes=["model-history-btn"])
                        model_history_btn5 = gr.Button(visible=False, variant="secondary", size="sm", elem_classes=["model-history-btn"])
                    
                with gr.Column(scale=2):
                    model_dropdown = gr.Dropdown(
                        choices=fetch_hf_models(),
                        label="🎯 Popular Models",
                        value="google/gemma-3-4b-it" if fetch_hf_models() else None,
                        allow_custom_value=True
                    )
                    
                    # Add a search function for the model dropdown
                    def update_model_choices(query):
                        """Update model choices based on search query"""
                        if query and len(query) > 2:  # Only search if query is long enough
                            search_results = search_all_models(query)
                            # Include the current query as a valid choice even if not in search results
                            if query not in search_results:
                                search_results.insert(0, query)
                            return gr.Dropdown(choices=search_results)
                        else:
                            # If no query or too short, show default HF models
                            default_models = fetch_hf_models()
                            return gr.Dropdown(choices=default_models)
                    
                    # Add Apply Model button and status display
                    with gr.Row():
                        apply_model_btn = gr.Button("Применить модель", variant="primary")
                        model_validation_status = gr.Textbox(
                            label="Статус модели",
                            interactive=False
                        )
                    
                    gr.Markdown("[Browse Hugging Face Models](https://huggingface.co/models)")
                    
                    # Function to handle model application
                    def apply_model(model_name):
                        """Validate and apply the selected model"""
                        status, color = validate_model(model_name)
                        # Return the status message
                        return status
                    
                    # Connect the Apply Model button
                    apply_model_btn.click(
                        fn=apply_model,
                        inputs=model_dropdown,
                        outputs=model_validation_status
                    )
              
                    with gr.Row():
                        with gr.Column(scale=2):
                            model_info = gr.Textbox(label="📊 Model Information", interactive=False, lines=8)

            def update_model_info_and_history(model_name):
                # Add to history only if it's a valid model selection
                if model_name and model_name != "":
                    # Check if model_name is in the current list or validate it on HuggingFace
                    current_models = fetch_hf_models() or []
                    if model_name in current_models or validate_hf_model(model_name):
                        history = add_to_model_history(model_name)
                    else:
                        history = model_history  # Use existing history
                else:
                    history = model_history  # Use existing history
                # Update model info
                info = load_model_info(model_name)
                # Update history buttons
                updates = []
                for i in range(5):
                    if i < len(history):
                        updates.extend([gr.Button(visible=True, value=history[i]), history[i]])
                    else:
                        updates.extend([gr.Button(visible=False), ""])
                return [info, gr.Column(visible=len(history) > 0)] + updates
            
            def select_history_model(model_name):
                return model_name
            
            model_dropdown.change(
                fn=update_model_info_and_history,
                inputs=model_dropdown,
                outputs=[model_info, model_history_container,
                        model_history_btn1, model_history_btn1,
                        model_history_btn2, model_history_btn2,
                        model_history_btn3, model_history_btn3,
                        model_history_btn4, model_history_btn4,
                        model_history_btn5, model_history_btn5]
            )
            
            # Connect history buttons
            model_history_btn1.click(fn=select_history_model, inputs=model_history_btn1, outputs=model_dropdown)
            model_history_btn2.click(fn=select_history_model, inputs=model_history_btn2, outputs=model_dropdown)
            model_history_btn3.click(fn=select_history_model, inputs=model_history_btn3, outputs=model_dropdown)
            model_history_btn4.click(fn=select_history_model, inputs=model_history_btn4, outputs=model_dropdown)
            model_history_btn5.click(fn=select_history_model, inputs=model_history_btn5, outputs=model_dropdown)
        
        # Dataset Preparation Tab
        with gr.TabItem("📂 Dataset Preparation"):
            gr.Markdown("## Prepare Your Dataset")
            
            with gr.Tabs():
                with gr.TabItem("📝 Create Dataset"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 💬 Conversation Builder")
                            conversation_builder = gr.Chatbot(
                                label="Build Conversation Template",
                                height=400
                            )
                            
                            with gr.Row():
                                user_msg = gr.Textbox(
                                    label="User Message",
                                    placeholder="Enter user message...",
                                    scale=3
                                )
                                add_user_btn = gr.Button("👤 Add User Message", scale=1)
                            
                            with gr.Row():
                                assistant_msg = gr.Textbox(
                                    label="Assistant Message",
                                    placeholder="Enter assistant message...",
                                    scale=3
                                )
                                add_assistant_btn = gr.Button("🤖 Add Assistant Message", scale=1)
                            
                            with gr.Row():
                                clear_conv_btn = gr.Button("🗑️ Clear Conversation")
                                save_template_btn = gr.Button("💾 Save Template")
                            
                            template_name = gr.Textbox(
                                label="Template Name",
                                placeholder="Enter template name..."
                            )
                        
                        with gr.Column():
                            gr.Markdown("### ⚙️ Dataset Generation")
                            system_prompt_input = gr.Textbox(
                                label="System Prompt",
                                lines=3,
                                placeholder="Enter system prompt for the model..."
                            )
                            
                            num_examples = gr.Number(
                                label="Number of Examples to Generate",
                                value=1,
                                precision=0
                            )
                            
                            
                            model_selector = gr.Dropdown(
                                choices=fetch_openrouter_models() or ["⚠️ OpenRouter not connected - Please set API key in Settings"],
                                label="Generation Model",
                                value=(fetch_openrouter_models()[0] if fetch_openrouter_models() else
                                      "⚠️ OpenRouter not connected - Please set API key in Settings"),
                                allow_custom_value=True
                            )
                            gr.Markdown("[Browse OpenRouter Models](https://openrouter.ai/models)")
                            output_filename = gr.Textbox(
                                label="Output Filename",
                                value="dataset.json"
                            )
                            
                            append_to_existing = gr.Checkbox(
                                label="Append to existing dataset",
                                value=False
                            )
                            
                            create_btn = gr.Button("🚀 Create Dataset", variant="primary", elem_classes=["main-action-btn"])
                            
                            gr.Markdown("### 📊 Dataset Creation")
                            dataset_output = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                            dataset_preview = gr.Dataframe(
                                label="Dataset Preview",
                                interactive=False
                            )
                
                with gr.TabItem("📁 Load Dataset"):
                    with gr.Row():
                        dataset_file = gr.File(label="📁 Upload Dataset File", file_types=[".json"])
                        with gr.Column():
                            load_dataset_btn = gr.Button("📥 Load Dataset")
                            dataset_info = gr.Dataframe(
                                label="📊 Dataset Information",
                                interactive=False
                            )
            
            # Event handlers for Dataset Preparation
            create_btn.click(
                fn=create_dataset_gradio,
                inputs=[conversation_builder, system_prompt_input, model_selector, output_filename, num_examples, append_to_existing],
                outputs=[dataset_output, dataset_preview]
            )
            
            load_dataset_btn.click(
                fn=load_dataset_preview,
                inputs=dataset_file,
                outputs=dataset_info
            )
            
            # Event handlers for Conversation Builder
            add_user_btn.click(
                fn=add_user_message,
                inputs=[conversation_builder, user_msg],
                outputs=[conversation_builder, user_msg]
            )
            
            add_assistant_btn.click(
                fn=add_assistant_message,
                inputs=[conversation_builder, assistant_msg],
                outputs=[conversation_builder, assistant_msg]
            )
            
            clear_conv_btn.click(
                fn=clear_conversation,
                inputs=None,
                outputs=[conversation_builder, user_msg, assistant_msg]
            )
            
            save_template_btn.click(
                fn=save_conversation_template,
                inputs=[conversation_builder, template_name],
                outputs=dataset_output
            )
        
        # Training Configuration Tab
        with gr.TabItem("⚙️ Training Configuration"):
            gr.Markdown("## Configure Training Parameters")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🔧 Primary Parameters")
                    lora_r = gr.Slider(
                        minimum=1, maximum=128, value=16, step=1,
                        label="Rank (r)"
                    )
                    lora_alpha = gr.Slider(
                        minimum=1, maximum=256, value=32, step=1,
                        label="Alpha"
                    )
                    num_epochs = gr.Slider(
                        minimum=1, maximum=50, value=8, step=1,
                        label="Number of Epochs"
                    )
                    batch_size = gr.Slider(
                        minimum=1, maximum=16, value=1, step=1,
                        label="Batch Size"
                    )
                
                with gr.Column():
                    with gr.Accordion("🔧 QLoRA Parameters", open=False):
                        lora_dropout = gr.Slider(
                            minimum=0.0, maximum=0.5, value=0.05, step=0.01,
                            label="Dropout"
                        )
                        target_modules = gr.Radio(
                            choices=["all-linear", "q_proj,v_proj", "k_proj,o_proj"],
                            value="all-linear",
                            label="Target Modules"
                        )
                    
                    with gr.Accordion("💾 Memory Optimization", open=False):
                        gradient_checkpointing = gr.Checkbox(
                            label="Gradient Checkpointing",
                            value=True
                        )
                        fp16 = gr.Checkbox(
                            label="FP16 Precision",
                            value=True
                        )
                        optim = gr.Dropdown(
                            choices=["paged_adamw_8bit", "adamw_torch", "adamw_hf"],
                            value="paged_adamw_8bit",
                            label="Optimizer",
                            visible=False
                        )
                    
                    with gr.Accordion("📊 Logging & Saving", open=False):
                        logging_steps = gr.Number(
                            label="Logging Steps",
                            value=20,
                            precision=0
                        )
                        save_steps = gr.Number(
                            label="Save Steps",
                            value=200,
                            precision=0
                        )
                        save_strategy = gr.Radio(
                            choices=["steps", "epoch"],
                            value="steps",
                            label="Save Strategy"
                        )
                        grad_accum = gr.Slider(
                            minimum=1, maximum=32, value=8, step=1,
                            label="Gradient Accumulation Steps"
                        )
                        learning_rate = gr.Number(
                            label="Learning Rate",
                            value=1e-4,
                            precision=6
                        )
            
            # Add some spacing
            gr.Markdown("")
        
        # Training Execution Tab
        with gr.TabItem("🏃 Training"):
            gr.Markdown("## Start Training Process")
            
            with gr.Row():
                train_btn = gr.Button("🚀 Start Training", variant="primary", elem_classes=["main-action-btn"])
                stop_btn = gr.Button("⏹️ Stop Training", variant="secondary")
            
            with gr.Row():
                training_progress = gr.Slider(
                    minimum=0, maximum=100, value=0,
                    label="Training Progress",
                    interactive=False
                )
                training_progress_label = gr.Label(value="0% done", label="Progress")
            
            with gr.Row():
                training_logs = gr.Code(
                    label="📋 Training Logs",
                    lines=20,
                    interactive=False
                )
            
            # Event handlers for Training
            train_btn.click(
                fn=start_training_wrapper,
                inputs=[model_dropdown, dataset_file, lora_r, lora_alpha, lora_dropout, target_modules, 
                       num_epochs, batch_size, grad_accum, learning_rate, gradient_checkpointing, fp16],
                outputs=training_logs
            )
            
            stop_btn.click(
                fn=stop_training,
                outputs=training_logs
            )
        
        # Model Conversion Tab
        with gr.TabItem("🔄 Model Conversion"):
            gr.Markdown("## Convert Model to GGUF and Quantize")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 GGUF Conversion")
                    quantization_type = gr.Dropdown(
                        choices=["Q4_K_M", "Q5_K_M", "Q8_0", "F16"],
                        value="Q4_K_M",
                        label="Quantization Type"
                    )
                    output_format = gr.Radio(
                        choices=["gguf", "bin"],
                        value="gguf",
                        label="Output Format"
                    )
                    model_path_input = gr.Textbox(
                        label="📁 Model Path",
                        placeholder="Path to your trained model"
                    )
                    convert_btn = gr.Button("🔨 Convert Model", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### 🐳 Ollama Integration")
                    ollama_model_name = gr.Textbox(
                        label="🏷️ Ollama Model Name",
                        placeholder="Enter model name for Ollama"
                    )
                    push_to_ollama_btn = gr.Button("📤 Push to Ollama", variant="secondary")
            
            with gr.Row():
                conversion_logs = gr.Textbox(
                    label="📋 Conversion Logs",
                    lines=10,
                    interactive=False
                )
            
            # Event handlers for Conversion
            convert_btn.click(
                fn=convert_to_gguf_wrapper,
                inputs=[model_path_input, quantization_type, output_format],
                outputs=conversion_logs
            )
            
            push_to_ollama_btn.click(
                fn=push_to_ollama_wrapper,
                inputs=[model_path_input, ollama_model_name],
                outputs=conversion_logs
            )
        
        # Template Management Tab
        with gr.TabItem("📝 Template Management"):
            gr.Markdown("## Chat Template Configuration")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🤖 Automatic Template")
                    auto_template_btn = gr.Button("🔍 Fetch Template from Model")
                    template_preview = gr.Textbox(
                        label="👁️ Template Preview",
                        lines=10,
                        interactive=False
                    )
                
                with gr.Column():
                    gr.Markdown("### ✏️ Custom Template")
                    custom_template = gr.Textbox(
                        label="📝 Custom Template",
                        lines=10,
                        placeholder="Enter your custom chat template here..."
                    )
                    save_template_btn = gr.Button("💾 Save Template")
                    template_list = gr.Dropdown(
                        choices=load_conversation_templates(),
                        label="💾 Available Templates",
                        interactive=True
                    )
                    refresh_templates_btn = gr.Button("🔄 Refresh Templates")
            
            with gr.Row():
                template_status = gr.Textbox(
                    label="📊 Template Status",
                    interactive=False
                )
            
            # Event handlers for Template Management
            auto_template_btn.click(
                fn=fetch_template_from_model_wrapper,
                inputs=model_dropdown,
                outputs=template_preview
            )
            
            save_template_btn.click(
                fn=save_custom_template_wrapper,
                inputs=custom_template,
                outputs=template_status
            )
            
            def refresh_templates():
                return gr.Dropdown(choices=load_conversation_templates())
            
            refresh_templates_btn.click(
                fn=refresh_templates,
                outputs=template_list
            )
            
            def load_selected_template(template_name):
                if not template_name:
                    return ""
                try:
                    templates_dir = "templates"
                    template_path = os.path.join(templates_dir, f"{template_name}.json")
                    with open(template_path, "r", encoding="utf-8") as f:
                        template_data = json.load(f)
                    # Convert to string format
                    template_str = json.dumps(template_data, ensure_ascii=False, indent=2)
                    return template_str
                except Exception as e:
                    return f"Error loading template: {str(e)}"
            
            template_list.change(
                fn=load_selected_template,
                inputs=template_list,
                outputs=custom_template
            )
        
        # Settings Tab
        with gr.TabItem("⚙️ Settings"):
            gr.Markdown("# Settings")
            
            with gr.Tabs():
                with gr.TabItem("🔑 API Keys"):
                    gr.Markdown("## API Key Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            openrouter_key_input = gr.Textbox(
                                label="OpenRouter API Key",
                                value=openrouter_key or "",  # Use loaded key or empty string
                                placeholder="Enter your OpenRouter API key",
                                type="password"
                            )
                            with gr.Row():
                                save_keys_btn = gr.Button("💾 Save API Keys", variant="primary")
                                validate_key_btn = gr.Button("🔍 Validate Key", variant="secondary")
                        
                        with gr.Column():
                            gr.Markdown("""
                            ### Where to get API keys:
                            - **OpenRouter**: [https://openrouter.ai/](https://openrouter.ai/)
                              - Required for dataset generation using large models
                            """)
                            key_status = gr.Markdown(f"**Key Status:** {'✅ Key saved' if openrouter_key else '❌ No key saved'}")
                    
                    keys_status = gr.Textbox(
                        label="Status",
                        interactive=False
                    )
                
                with gr.TabItem("🖥️ System Info"):
                    gr.Markdown("## System Information")
                    system_info = gr.Markdown()
                    refresh_system_btn = gr.Button("🔄 Refresh System Info")
                
                with gr.TabItem("❓ Help"):
                    gr.Markdown("## Help & Documentation")
                    help_info = gr.Markdown()
                    refresh_help_btn = gr.Button("🔄 Refresh Help")
            
            # Event handlers for Settings
            def save_and_update_status(key_input):
                # Save the API key
                save_result = save_api_keys(key_input)
                # Update the key status display
                status_text = f"**Key Status:** {'✅ Key saved' if openrouter_key else '❌ No key saved'}"
                return save_result, gr.Markdown(status_text)
            
            def refresh_model_selector():
                """Refresh the model selector with updated API key"""
                models = fetch_openrouter_models() or ["⚠️ OpenRouter not connected - Please set API key in Settings"]
                first_model = models[0] if models else "⚠️ OpenRouter not connected - Please set API key in Settings"
                return gr.Dropdown(choices=models, value=first_model)
            
            save_keys_btn.click(
                fn=save_and_update_status,
                inputs=openrouter_key_input,
                outputs=[keys_status, key_status]
            ).then(
                fn=refresh_model_selector,
                inputs=None,
                outputs=model_selector
            )
            
            validate_key_btn.click(
                fn=validate_openrouter_key,
                inputs=openrouter_key_input,
                outputs=keys_status
            )
            
            refresh_system_btn.click(
                fn=get_system_info,
                outputs=system_info
            )
            
            refresh_help_btn.click(
                fn=get_help_info,
                outputs=help_info
            )
            
            # Initialize system info and help on load
            demo.load(get_system_info, None, system_info)
            demo.load(get_help_info, None, help_info)

# Launch the app
if __name__ == "__main__":
    demo.launch()