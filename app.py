#%%
import os
from typing import Optional, List, Dict, Any

import httpx
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import dotenv

from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message import ChatCompletionMessage
# import openai
from langchain_openai import ChatOpenAI

# Load environment variables
dotenv.load_dotenv()

# For LangChain integration:
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "nebula"

#%%

###############################################################################
# Settings and Configuration
###############################################################################

class Settings(BaseSettings):
    # Your Nebula secret key; used for authentication to Thirdweb's Nebula API.
    nebula_secret_key: str = os.getenv("THIRDWEB_SECRET_KEY")
    # Base URL for all Nebula API calls.
    nebula_base_url: str = "https://nebula-api.thirdweb.com"
    # Optional separate API key for OpenAI (if desired); if not provided, the Nebula key is used.
    openai_api_key: Optional[str] = None
    # Additional Thirdweb configuration
    next_public_thirdweb_client_id: Optional[str] = None
    thirdweb_secret_key: Optional[str] = None

    model_config = {
        'env_file': '.env',
        'extra': 'allow'  # This allows extra fields from environment variables
    }

settings = Settings()

# Configure the OpenAI library to use Nebula's API.
# This lets you call openai.ChatCompletion.create (or its async variant) while
# # automatically sending requests to Nebula's backend.
# openai.api_key = settings.nebula_secret_key
# openai.api_base = settings.nebula_base_url

#%%
###############################################################################
# Request Models
###############################################################################

# Model for wrapping the Nebula /chat endpoint
class NebulaChatRequest(BaseModel):
    message: str
    stream: Optional[bool] = False
    session_id: Optional[str] = None
    execute_config: Optional[Dict[str, Any]] = None
    context_filter: Optional[Dict[str, Any]] = None

# Model for the OpenAI-compatible chat completions endpoint
class OpenAIChatRequest(BaseModel):
    # Model to use; default is "t0" (as in the example)
    model: Optional[str] = "t0"
    # List of messages following the standard OpenAI format
    messages: List[Dict[str, str]]
    stream: Optional[bool] = False
    # Additional parameters to include in the request body (e.g., context filters)
    extra_body: Optional[Dict[str, Any]] = None

# Model for testing LangChain chat integration
class LangChainChatRequest(BaseModel):
    # A list of messages, each message should include a "role" and a "content"
    messages: List[Dict[str, str]]


class NebulaContext(BaseModel):
    session_id: str
    wallet_address: str
    chain_ids: Optional[List[str]] = None

class NebulaChatResponse(BaseModel):
    id: str
    message: ChatCompletionMessage
    usage: CompletionUsage
    context: NebulaContext

###############################################################################
# FastAPI Application Setup
###############################################################################

app = FastAPI(title="Nebula API Wrapper via FastAPI")

###############################################################################
# Endpoints
###############################################################################

@app.get("/")
async def read_root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Welcome to the Nebula API Wrapper via FastAPI"}

# @app.post("/nebula-chat")
# async def nebula_chat(request_body: NebulaChatRequest):
#     """
#     This endpoint wraps the Nebula /chat endpoint directly.
#     It forwards the request to Nebula with the proper authentication.
#     """
#     url = f"{settings.nebula_base_url}/chat"
#     headers = {
#         "Content-Type": "application/json",
#         "x-secret-key": settings.nebula_secret_key,
#     }
#     payload = request_body.dict(exclude_none=True)

#     async with httpx.AsyncClient() as client:
#         response = await client.post(url, json=payload, headers=headers)

#     if response.status_code != 200:
#         # You may want to log response.text here for debugging.
#         raise HTTPException(status_code=response.status_code, detail=response.text)

#     return response.json()

@app.post("/chat")
async def openai_chat(request_body: OpenAIChatRequest) -> NebulaChatResponse:
    """
    This endpoint demonstrates using the OpenAI Python library (configured to use
    Nebula as the backend) to process chat completions.
    
    Example Request Body:
    {
      "model": "t0",
      "messages": [{"role": "user", "content": "Hello Nebula!"}],
      "stream": false,
      "extra_body": {"context": {"wallet_address": "0x514C373696cCc04C34600ED42D8858A370ad74Cb"}}
    }
    """
    try:
        # Prepare parameters for the OpenAI Chat Completion request.
        params = {
            "model": request_body.model,
            "messages": request_body.messages,
            "stream": request_body.stream,
        }

        client = OpenAI(
            base_url=settings.nebula_base_url,
            api_key=settings.nebula_secret_key,
        )

        chat_completion = client.chat.completions.create(
            model=request_body.model,
            messages=request_body.messages,
            stream=request_body.stream,
            extra_body=request_body.extra_body
        )

        
        # # Merge any extra parameters (e.g., context filters) into the payload.
        # if request_body.extra_body:
        #     params.update(request_body.extra_body)

        # Use the async version if streaming is not required.
        # (For streaming responses, more work is needed to yield events.)
        # result = await OpenAI.ChatCompletion.acreate(**params)
        return NebulaChatResponse(
            id=chat_completion.id,
            message=chat_completion.choices[-1].message,
            usage=chat_completion.usage,
            context=NebulaContext(**chat_completion.context)
        )   
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/langchain-chat")
# async def langchain_chat(request_body: LangChainChatRequest):
#     """
#     This endpoint demonstrates using LangChain's ChatOpenAI integration configured to use
#     Nebula as the OpenAI API backend.

#     Example Request Body:
#     {
#       "messages": [
#          {"role": "system", "content": "You are a helpful assistant."},
#          {"role": "user", "content": "Hello Nebula via LangChain!"}
#       ]
#     }
#     """
#     try:
#         # Use the provided OPENAI_API_KEY if available, otherwise fallback to the Nebula key
#         api_key = settings.openai_api_key or settings.nebula_secret_key

#         # Instantiate the LangChain ChatOpenAI model with custom parameters
#         llm = ChatOpenAI(
#             model_name="gpt-4",  # Using standard model name
#             temperature=0,
#             openai_api_key=api_key,
#             openai_api_base=settings.nebula_base_url,
#         )

#         # LangChain's ChatOpenAI is synchronous
#         # Run it in a threadpool to avoid blocking the async event loop
#         response = await run_in_threadpool(llm, request_body.messages)

#         return {"response": response}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# %%


#%%
# import uuid
# session_id = uuid.uuid4()

# # %%
# openai_request = OpenAIChatRequest(
#     messages=[
#         {"role": "user", "content": "Whats my name?."}
#     ], 
#     extra_body={
#         "context": {
#             "session_id": "2e19b6c4-f099-4b7f-af44-6b0b471885c7",
#             "wallet_address": "0x514C373696cCc04C34600ED42D8858A370ad74Cb",
#             "chain_ids": ["80002", "8453"],
#         }
#     }
# )
# # # %%
# result = await openai_chat(openai_request)
# # # %%

# %%
