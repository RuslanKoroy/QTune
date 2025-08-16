import os
import subprocess
import sys
from pathlib import Path

def check_llama_cpp_installed():
    try:
        # Try to run llama.cpp help command
        result = subprocess.run(["llama-quantize", "--help"], 
                              capture_output=True, text=True, timeout=10)
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_to_gguf(model_path, output_path, quantization_type="Q4_K_M"):
    try:
        # Check if model path exists
        if not os.path.exists(model_path):
            return False, f"Model path does not exist: {model_path}"
        
        # Check if llama.cpp tools are available
        if not check_llama_cpp_installed():
            return False, "llama.cpp tools not found. Please install llama.cpp first."
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Convert model to GGUF format
        
        # Check if the model is in a format that needs conversion to ggml first
        model_extension = os.path.splitext(model_path)[1].lower()
        
        if model_extension in ['.bin', '.pt', '.pth']:
            # Convert to ggml first
            cmd = ["python", "convert-hf-to-gguf.py", model_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, f"Error converting model to ggml: {result.stderr}"
            
            # The conversion script typically outputs to the same directory
            # with a .gguf extension
            ggml_path = os.path.splitext(model_path)[0] + ".gguf"
        else:
            ggml_path = model_path
            
        # Quantize the model
        cmd = ["llama-quantize", ggml_path, output_path, quantization_type]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, f"Error during quantization: {result.stderr}"
            
        message = f"Successfully converted {model_path} to GGUF format with {quantization_type} quantization.\n"
        message += f"Output saved to: {output_path}"
        
        return True, message
        
    except Exception as e:
        return False, f"Error during conversion: {str(e)}"

def push_to_ollama(model_path, model_name):
    try:
        # Check if Ollama is installed and running
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True, timeout=10)
            ollama_available = True
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            ollama_available = False
        
        if not ollama_available:
            return False, "Ollama not found. Please install and start Ollama first."
        
        # Check if model file exists
        if not os.path.exists(model_path):
            return False, f"Model file not found: {model_path}"
        
        # Create a temporary model file for Ollama
        model_file_content = f"""FROM {model_path}
PARAMETER temperature 0.7
PARAMETER stop Result
PARAMETER stop Human
PARAMETER stop ###
"""
        
        # Create a temporary model file
        temp_model_file = f"/tmp/{model_name}.modelfile"
        with open(temp_model_file, 'w') as f:
            f.write(model_file_content)
            
        # Push the model to Ollama
        cmd = ["ollama", "create", model_name, "-f", temp_model_file]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, f"Error creating Ollama model: {result.stderr}"
            
        message = f"Successfully pushed model to Ollama as '{model_name}'.\n"
        message += "You can now use the model with: ollama run " + model_name
        
        return True, message
        
    except Exception as e:
        return False, f"Error during Ollama push: {str(e)}"

