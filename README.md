# PyNebula

This repository contains a FastAPI application that acts as a wrapper around Thirdweb's Nebula API. It exposes endpoints for:

- **Direct Nebula API Chat:** Forward chat messages and transaction requests directly to Nebula.
- **OpenAI-Compatible Chat Completions:** Use the OpenAI Python SDK (configured to use Nebula as the backend) to process chat messages.
- **LangChain Integration:** Leverage LangChain’s ChatOpenAI integration to create context-aware chat interactions.

Each endpoint demonstrates a different method of interacting with Nebula, allowing you to choose the one that best suits your application's needs.

---

## Features

- **Direct Nebula Chat Endpoint:**  
  Forward requests to Nebula’s `/chat` endpoint with proper authentication and payload formatting.

- **OpenAI Chat Completions Endpoint:**  
  Uses the OpenAI Python library configured with Nebula’s base URL, allowing you to call chat completions in a familiar way.

- **LangChain Chat Integration:**  
  Integrates with LangChain’s ChatOpenAI to leverage high-level abstractions (prompt chaining, context management, etc.) for conversational AI.

- **Environment-Based Configuration:**  
  Configure your Nebula secret key and (optionally) an OpenAI API key using environment variables.

---

## Requirements

- Python 3.7+
- [FastAPI](https://fastapi.tiangolo.com/)
- [httpx](https://www.python-httpx.org/)
- [openai](https://pypi.org/project/openai/)
- [langchain](https://github.com/hwchase17/langchain)
- [uvicorn](https://www.uvicorn.org/)
- [python-dotenv](https://github.com/theskumar/python-dotenv)

Install all required dependencies with:

```bash
pip install fastapi uvicorn httpx openai langchain python-dotenv