# ScriptumAI Complete Installation Guide

This guide will walk you through the entire process of setting up ScriptumAI, including all prerequisites and necessary components.

## Table of Contents
- [ScriptumAI Complete Installation Guide](#scriptumai-complete-installation-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Install Python 12](#1-install-python-12)
  - [2. Install CUDA Toolkit (for GPU use)](#2-install-cuda-toolkit-for-gpu-use)
  - [3. Install Ollama](#3-install-ollama)
    - [For macOS and Linux:](#for-macos-and-linux)
    - [For Windows:](#for-windows)
  - [4. Pull Required Models](#4-pull-required-models)
  - [5. Set Up ScriptumAI](#5-set-up-scriptumai)
  - [Troubleshooting](#troubleshooting)

## 1. Install Python 12

First, check if Python 12 is already installed:

```bash
python --version
```

If Python 12 is not installed:

1. Visit https://www.python.org/downloads/
2. Download Python 12.x.x
3. Run the installer, ensuring you check "Add Python to PATH"
4. Verify installation by running `python --version` in a new terminal

## 2. Install CUDA Toolkit (for GPU use)

If you have a compatible NVIDIA GPU and want to use it for accelerated processing:

1. Visit https://developer.nvidia.com/cuda-downloads
2. Select your operating system and follow the installation instructions
3. Verify installation by running `nvcc --version` in a terminal

## 3. Install Ollama

### For macOS and Linux:

```bash
curl https://ollama.ai/install.sh | sh
```

### For Windows:

1. Visit https://ollama.com/download/OllamaSetup.exe
2. Download and run the latest Windows installer

Verify installation:

```bash
ollama --version
```

## 4. Pull Required Models

Pull the necessary models for ScriptumAI:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify models are installed:

```bash
ollama list
```

## 5. Set Up ScriptumAI

1. Clone the ScriptumAI repository (replace with actual repository URL):

```bash
git clone https://github.com/yourusername/ScriptumAI.git
cd ScriptumAI
```

2. Create a virtual environment:

```bash
python -m venv scriptum python=3.12
```
Or
```bash
python3 -m venv scriptum pyhton=3.12
```

3. Activate the virtual environment:

- On Windows:
  ```
  scriptum\Scripts\activate
  ```
- On macOS and Linux:
  ```
  source scriptum/bin/activate
  ```

4. Install requirements:

```bash
pip install -r requirements.txt
```

 - MacOS 
```bash
brew install libmagic
```
```
- Reboot the computer.

## 6. Run ScriptumAI

1. In one terminal, ensure you're in the ScriptumAI directory and your virtual environment is activated, then run:

```bash
python api.py
```

2. Open a separate terminal, navigate to the ScriptumAI directory, activate the virtual environment, and run:

```bash
streamlit run app.py
```

This will start the ScriptumAI application. You can access the user interface by opening a web browser and navigating to the URL provided by Streamlit (typically http://localhost:8501).

## Troubleshooting

- If you encounter any "command not found" errors, ensure the relevant tool is correctly installed and added to your system's PATH.
- For GPU-related issues, make sure your NVIDIA drivers are up to date and compatible with the installed CUDA Toolkit.
- If you face problems with Ollama or model downloads, check your internet connection and firewall settings.
- For any Python package installation issues, ensure you're using the correct version of pip within your virtual environment.

If you continue to experience problems, please refer to the official documentation for each component.
