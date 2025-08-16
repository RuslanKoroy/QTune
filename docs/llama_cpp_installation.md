# Installing llama.cpp

To install llama.cpp for GGUF conversion, follow these steps:

## Prerequisites
- Git
- CMake (for building from source)
- A C++ compiler (GCC or Clang)

## Installation Steps

### 1. Clone the llama.cpp repository:
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
```

### 2. Compile the tools:
```bash
make
```

### 3. Add the tools to your PATH:
```bash
export PATH=$(pwd):$PATH
```

### 4. Verify installation:
```bash
llama-quantize --help
```

## Alternative Installation Methods

### Using Package Managers

#### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install llama.cpp
```

#### On macOS with Homebrew:
```bash
brew install llama.cpp
```

### Using Pre-compiled Binaries

You can download pre-compiled binaries from the [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases).

## Required Tools

The following tools from llama.cpp are used by this converter:
- `convert-hf-to-gguf.py` - For converting Hugging Face models to GGUF format
- `llama-quantize` - For quantizing GGUF models

Make sure these tools are available in your PATH after installation.