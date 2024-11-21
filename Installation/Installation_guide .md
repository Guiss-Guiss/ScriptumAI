# ScriptumAI | Complete Installation Guide

This guide will walk you through the entire process of setting up ScriptumAI, including all prerequisites and necessary components.

## Table of Contents
1. [Install Python 11](#1-install-python-11)
2. [Install Ollama](#2-install-ollama)
4. [Pull Required Models](#4-pull-required-models)
5. [Set Up ScriptumAI](#5-set-up-scriptum-ai)
6. [Run ScriptumAI](#6-run-scriptum-ai)

## 1. Install Python 11

First, check if Python 11 is already installed:

```bash
python --version
```

If Python 11 is not installed:

1. Visit https://www.python.org/downloads/
2. Download Python 11.x.x
3. Run the installer, ensuring you check "Add Python to PATH"
4. Verify installation by running `python --version` in a new terminal

## 2. Install Ollama

### For macOS and Linux:

```bash
curl https://ollama.ai/install.sh | sh
```

### For Windows:

1. Visit https://ollama.com/download/windows
2. Download and run the latest Windows installer
Verify installation:

```bash
ollama --version
```

## 3. Pull Required Models

Pull the necessary models for ScriptumAI :

```bash
ollama pull llama3.1:latest
ollama pull nomic-embed-text
```

Verify models are installed:

```bash
ollama list
```

##4. Set Up ScriptumAI

1. Clone the ScriptumAI repository :

```bash
git clone https://github.com/Guiss-Guiss/ScriptumAI-CPU-.git
cd ScriptumAI
```

2. Create a virtual environment:

```bash
python -m venv scriptum python=3.11

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

## 5. Run ScriptumAI


2. Open a terminal, navigate to the ScriptumAICPU directory, activate the virtual environment, and run:

```bash
streamlit run app.py
```

This will start the ScriptumAI application. You can access the user interface by opening a web browser and navigating to the URL provided by Streamlit (typically http://localhost:8501).

## Troubleshooting

- If you encounter any "command not found" errors, ensure the relevant tool is correctly installed and added to your system's PATH.
- If you face problems with Ollama or model downloads, check your internet connection and firewall settings.
- For any Python package installation issues, ensure you're using the correct version of pip within your virtual environment.

If you continue to experience problems, please refer to the official documentation for each component.
