# main.py
#%%
import os
import json
import asyncio
from typing import Optional, List, Dict, Any

import httpx
import openai
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Nebula API base URL and thirdweb secret key.
NEBULA_API_BASE_URL = "https://nebula-api.thirdweb.com"
THIRDWEB_SECRET_KEY = os.getenv("THIRDWEB_SECRET_KEY")
# Optionally, you can set OPENAI_API_KEY for LangChain usage
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", THIRDWEB_SECRET_KEY)

# Configure the OpenAI Python library to use the Nebula API.
openai.api_base = NEBULA_API_BASE_URL
openai.api_key = THIRDWEB_SECRET_KEY

#%%
# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ExecuteConfig(BaseModel):
    mode: str
    signer_wallet_address: str

class ContextFilter(BaseModel):
    chain_ids: Optional[List[int]] = None
    contract_addresses: Optional[List[str]] = None
    wallet_addresses: Optional[List[str]] = None

class ChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    execute_config: Optional[ExecuteConfig] = None
    context_filter: Optional[ContextFilter] = None

class SessionRequest(BaseModel):
    title: Optional[str] = None
    execute_config: Optional[ExecuteConfig] = None
    context_filter: Optional[ContextFilter] = None

# Models for OpenAI endpoints

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False
    # extra_body can include additional parameters like context, etc.
    extra_body: Optional[Dict[str, Any]] = {}

class LangChainChatRequest(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 0.0
    messages: List[Dict[str, str]]

# ---------------------------------------------------------------------------
# FastAPI App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Thirdweb Nebula API Wrapper with OpenAI & LangChain",
    description="A FastAPI app that wraps the thirdweb Nebula API endpoints and integrates OpenAI and LangChain functionality.",
    version="1.1.0",
)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def get_auth_headers() -> Dict[str, str]:
    """
    Return headers required for authenticating with the Nebula API.
    """
    return {
        "x-secret-key": THIRDWEB_SECRET_KEY,
        "Content-Type": "application/json",
    }

async def event_stream_generator(response: httpx.Response):
    """
    Asynchronously yield lines from a streaming response.
    """
    async for line in response.aiter_lines():
        if line:
            yield f"{line}\n"

# ---------------------------------------------------------------------------
# Endpoints for Nebula /chat, /execute, and session management
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    """
    Proxies the /chat endpoint from the Nebula API.
    """
    url = f"{NEBULA_API_BASE_URL}/chat"
    headers = get_auth_headers()
    payload = chat_request.dict(exclude_none=True)
    
    async with httpx.AsyncClient() as client:
        if chat_request.stream:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        content = await resp.aread()
                        raise HTTPException(status_code=resp.status_code, detail=content.decode())
                    return StreamingResponse(event_stream_generator(resp), media_type="text/event-stream")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")
        else:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

@app.post("/execute")
async def execute_endpoint(chat_request: ChatRequest):
    """
    Proxies the /execute endpoint from the Nebula API.
    """
    url = f"{NEBULA_API_BASE_URL}/execute"
    headers = get_auth_headers()
    payload = chat_request.dict(exclude_none=True)

    async with httpx.AsyncClient() as client:
        if chat_request.stream:
            try:
                async with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code != 200:
                        content = await resp.aread()
                        raise HTTPException(status_code=resp.status_code, detail=content.decode())
                    return StreamingResponse(event_stream_generator(resp), media_type="text/event-stream")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")
        else:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code != 200:
                    raise HTTPException(status_code=resp.status_code, detail=resp.text)
                return resp.json()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Request error: {str(e)}")

@app.post("/session")
async def create_session(session_request: SessionRequest):
    """
    Creates a new session using the Nebula API.
    """
    url = f"{NEBULA_API_BASE_URL}/session"
    headers = get_auth_headers()
    payload = session_request.dict(exclude_none=True)

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, headers=headers, json=payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Session creation error: {str(e)}")

@app.get("/session/list")
async def list_sessions():
    """
    Lists sessions from the Nebula API.
    """
    url = f"{NEBULA_API_BASE_URL}/session/list"
    headers = get_auth_headers()

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"List sessions error: {str(e)}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Retrieves details of a specific session.
    """
    url = f"{NEBULA_API_BASE_URL}/session/{session_id}"
    headers = get_auth_headers()

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Get session error: {str(e)}")

# ---------------------------------------------------------------------------
# New Endpoints for OpenAI-compatible and LangChain integration
# ---------------------------------------------------------------------------

@app.post("/openai/chat_completion")
async def openai_chat_completion(request_data: OpenAIChatCompletionRequest):
    """
    Uses the OpenAI Chat Completion API via the Nebula API.
    
    This endpoint calls openai.ChatCompletion.create with a custom base URL (set to Nebula)
    and supports streaming responses if specified.
    """
    try:
        # When stream=True, openai.ChatCompletion.create returns a generator.
        result = openai.ChatCompletion.create(
            model=request_data.model,
            messages=request_data.messages,
            stream=request_data.stream,
            **(request_data.extra_body or {})
        )
        # If streaming, we need to yield events as Server-Sent Events.
        if request_data.stream:
            def generator():
                try:
                    for event in result:
                        # You can format the streamed event as needed.
                        yield f"data: {json.dumps(event)}\n\n"
                except Exception as stream_e:
                    yield f"data: {json.dumps({'error': str(stream_e)})}\n\n"
            return StreamingResponse(generator(), media_type="text/event-stream")
        else:
            # For non-streaming, return the full result.
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/openai/models")
async def openai_models():
    """
    Retrieves the list of models from the Nebula API using the OpenAI Models endpoint.
    """
    try:
        models = openai.Model.list()
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/openai/langchain_chat")
async def langchain_chat(request_data: LangChainChatRequest):
    """
    Demonstrates LangChain integration with the Nebula API.
    
    This endpoint uses LangChain's ChatOpenAI class to generate a chat completion.
    It uses the Nebula API as the base URL.
    """
    try:
        # Import ChatOpenAI from LangChain.
        from langchain.chat_models import ChatOpenAI
        
        # Instantiate the LangChain chat model with our custom settings.
        llm = ChatOpenAI(
            model_name=request_data.model,
            temperature=request_data.temperature,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=NEBULA_API_BASE_URL,  # custom base URL for Nebula
        )
        # Invoke the model with the provided messages.
        # (LangChain expects a list of message dicts.)
        response = llm(request_data.messages)
        # The response object typically has a "content" attribute.
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))