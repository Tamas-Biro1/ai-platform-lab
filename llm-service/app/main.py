import json
import logging
import os
import sys
from time import perf_counter
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")


class IncidentSummary(BaseModel):
    summary: str = Field(max_length=500)
    severity: Literal["low", "medium", "high", "critical"]
    service: str


class LogIn(BaseModel):
    log: str = Field(min_length=1, max_length=20_000)

SYSTEM = (
    "You are an SRE assistant. Only respond any objects that correspond to this JSON-schema:\n"
    + json.dumps(IncidentSummary.model_json_schema())
)

app = FastAPI()
start_time: float


log = logging.getLogger("my-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))

def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.getenv("LLM_API_KEY", "ollama"))

ClientDep = Annotated[AsyncOpenAI, Depends(get_client)]

@app.post("/summarize", response_model=IncidentSummary)
async def summarize(body: LogIn, client: AsyncOpenAI = ClientDep) -> IncidentSummary:
    for _ in range(3):
        resp = await client.chat.completions.create(
            model=MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": body.log},
            ],
        )
        try:
            return IncidentSummary.model_validate_json(resp.choices[0].message.content or "")
        except ValidationError:
            continue  # model gave wrong JSON, retry...
    raise HTTPException(status_code=502, detail="The model did not return a valid response after 3 attempts")


# implementation with closures
#@app.post("/chat")
#async def chat(body: LogIn, client: AsyncOpenAI = Depends(get_client)):
#    async def events():
#        stream = await client.chat.completions.create(
#            model=MODEL, stream=True,
#            messages=[{"role": "user", "content": body.log}],
#        )
#        async for chunk in stream:
#            text = chunk.choices[0].delta.content or ""
#            if text:
#                yield f"data: {json.dumps({'text': text})}\n\n"
#        yield "data: [DONE]\n\n"
#
#    return StreamingResponse(events(), media_type="text/event-stream")

def measure_and_done() -> str:
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    log.info(f"elapsed time: {elapsed_time}\n\n")
    return "data: [DONE]\n\n"


async def events(client: AsyncOpenAI, message: str):
        stream = await client.chat.completions.create(
            model=MODEL, stream=True,
            messages=[{"role": "user", "content": message}],
            stream_options={"include_usage": True}
        )
        async for chunk in stream:
            usage = chunk.usage
            if usage:
                yield f"event: usage\ndata: {json.dumps(usage.model_dump())}\n\n"
                continue
            text = chunk.choices[0].delta.content or ""
            if text:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield measure_and_done()

@app.post("/chat")
async def chat(body: LogIn, client: AsyncOpenAI = ClientDep):
    global start_time
    start_time = perf_counter()
    return StreamingResponse(events(client, body.log), media_type="text/event-stream")

