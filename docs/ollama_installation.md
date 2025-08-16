# Installing Ollama

To install Ollama, follow these steps:

## Installation Steps

### 1. For Linux/Mac:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. For Windows:
Download the installer from [Ollama Setup.exe](https://ollama.com/download/OllamaSetup.exe)

### 3. Start Ollama:
```bash
ollama serve
```

### 4. Verify installation:
```bash
ollama --version
```

## System Requirements

- **Linux**: Most modern Linux distributions
- **macOS**: macOS 11+ (Big Sur)
- **Windows**: Windows 10 or later

## GPU Support

Ollama supports GPU acceleration on the following platforms:

### NVIDIA GPUs
- Install NVIDIA drivers
- CUDA support is included with Ollama

### AMD GPUs
- Install ROCm drivers
- AMD GPU support is experimental

### Apple Silicon
- Native support for M1/M2/M3 chips
- No additional drivers required

## Common Issues and Troubleshooting

### Permission Denied Errors
If you encounter permission errors, try:
```bash
sudo curl -fsSL https://ollama.com/install.sh | sh
```

### Service Not Starting
If Ollama service doesn't start automatically:
```bash
sudo systemctl start ollama
sudo systemctl enable ollama
```

### Model Pull Issues
If you have issues pulling models:
```bash
# Check if the service is running
ollama list

# Try pulling a model manually
ollama pull llama3
```

## Useful Commands

- `ollama list` - List downloaded models
- `ollama run <model>` - Run a model
- `ollama ps` - Show running models
- `ollama rm <model>` - Remove a model