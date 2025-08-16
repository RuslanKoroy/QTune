## Help & Documentation

### Getting Started
1. Set your API keys in the Settings tab
2. Select a model from the Model Selection tab
3. Prepare your dataset in the Dataset Preparation tab
4. Configure training parameters in the Training Configuration tab
5. Start training in the Training tab
6. Convert your model in the Model Conversion tab

### API Keys
- **OpenRouter API Key**: Required for dataset generation using large models
  - Get one at https://openrouter.ai/

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU with at least 8GB VRAM (recommended)
- At least 16GB system RAM

### Troubleshooting
- If you encounter memory errors, reduce batch size or enable gradient checkpointing
- For slow training, ensure you're using a CUDA-compatible GPU
- If models fail to load, check your internet connection

### Tips for Best Results
- Use high-quality, diverse datasets for training
- Start with smaller models for testing
- Monitor VRAM usage during training
- Experiment with different QLoRA parameters