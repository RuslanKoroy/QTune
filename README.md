# QTune Documentation

![Banner](docs/banner.jpg)

## Overview

QTune is a comprehensive web application for fine-tuning language models on consumer GPUs with as little as 8GB of VRAM. Built with Gradio, it provides an intuitive interface for the entire fine-tuning workflow.

## Features

### 1. Model Selection
- Automatically fetch models from Hugging Face
- Detailed model information including VRAM requirements
- Support for popular models like Gemma, Llama, Mistral, and more
- Manual model entry for models not in the list
- Automatic validation of manually entered models on HuggingFace
- Links to model repositories (Hugging Face and OpenRouter)

### 2. Dataset Preparation
- Create datasets using larger models via OpenRouter API
- Upload your own datasets in JSON format
- Dataset preview and validation
- Manual model entry for generation models not in the list

### 3. Training Configuration
- Fine-tune QLoRA parameters (rank, alpha, dropout, target modules)
- Configure training parameters (epochs, batch size, learning rate)
- Memory optimization settings for 8GB VRAM GPUs

### 4. Training Execution
- Start and monitor training process
- Real-time training logs
- Progress tracking

### 5. Model Conversion
- Convert models to GGUF format
- Multiple quantization options (Q4_K_M, Q5_K_M, Q8_0, F16)
- Integration with Ollama for easy deployment

### 6. Template Management
- Automatically fetch chat templates from models
- Create and save custom templates
- Apply templates to datasets

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU with at least 8GB VRAM (recommended)
- At least 16GB system RAM

### Quick Installation
```bash
# Clone the repository
git clone <repository-url>
cd qtune

# Install dependencies
pip install -r requirements.txt

# For GGUF conversion and Ollama integration (optional)
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
export PATH=$(pwd):$PATH

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

## Usage

### Starting the Application
```bash
python app.py
```

The application will start and provide a local URL to access the web interface.

### Workflow

1. **Select a Model**: Choose from popular models on Hugging Face
2. **Prepare Dataset**: Either generate a dataset using larger models or upload your own
3. **Configure Training**: Adjust QLoRA and training parameters
4. **Start Training**: Begin the fine-tuning process
5. **Convert Model**: Export to GGUF format for deployment
6. **Deploy**: Push to Ollama for easy inference

## API Keys

### OpenRouter API Key
To use the dataset generation features, you need an OpenRouter API key:
1. Sign up at https://openrouter.ai/
2. Get your API key from the dashboard
3. Enter the key in the Settings tab of the application
   - The key will be saved automatically to api_keys.json
   - It will be loaded automatically when the application starts

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed correctly
2. **CUDA Issues**: Ensure you have the correct CUDA version for your PyTorch installation
3. **Memory Errors**: Reduce batch size or enable gradient checkpointing
4. **Model Loading Issues**: Check internet connection and Hugging Face credentials

### Compatibility Notes
- The application is optimized for 8GB VRAM consumer GPUs
- Training on CPU is possible but will be very slow
- Some features require additional tools (llama.cpp, Ollama) to be installed separately
