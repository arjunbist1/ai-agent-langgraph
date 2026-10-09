

# AI Agent using LangGraph

A simple AI agent project built with Python, LangChain,
LangGraph and Google Gemini.

## Features

- Classifies questions into coding, weather and Google Search
- Routes questions to the relevant agent
- Uses Gemini for coding-related answers
- Uses Google Serper for web search
- Includes a mock weather tool for demonstration

## Setup

1. Clone the repository.
2. Create a virtual environment.
3. Install the requirements.
4. Configure your API keys in a local `.env` file.
5. Run the application.

## Installation

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file using `.env.example` as a template.
Add your own GOOGLE_API_KEY and SERPER_API_KEY.

## Run

```bash
python main.py
```

## Technologies

Python, LangChain, LangGraph, Google Gemini,
Google Serper, Pydantic