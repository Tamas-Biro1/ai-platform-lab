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

log = logging.getLogger("my-logger")
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler(sys.stdout))


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.getenv("LLM_API_KEY", "ollama"))


ClientDep = Annotated[AsyncOpenAI, Depends(get_client)]


@app.post("/summarize", response_model=IncidentSummary)
async def summarize(body: LogIn, client: ClientDep) -> IncidentSummary:
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
            return IncidentSummary.model_validate_json(
                resp.choices[0].message.content or ""
            )
        except ValidationError:
            continue  # model gave wrong JSON, retry...
    raise HTTPException(
        status_code=502,
        detail="The model did not return a valid response after 3 attempts",
    )


# generator implementation with closures
# @app.post("/chat")
# async def chat(body: LogIn, client: AsyncOpenAI = Depends(get_client)):
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


def measure_and_done(start_time: float, ttft: float, token_count: int) -> str:
    end_time = perf_counter()
    total = end_time - start_time
    log.info(
        json.dumps(
            {'total_ms': round(total * 1000, 1) ,
             'ttft_ms': round(ttft * 1000) if ttft is not None else None,
             'token_count': token_count
            }))
    return "data: [DONE]\n\n"


# generator function
async def events(client: AsyncOpenAI, message: str, request_starttime: float):
    # 'chunk' instead of 'token'
    # vLLM server might merge more
    # tokens into a chunk to send
    # rough estimation for the usage (include_usage is better)
    chunk_count = 0
    ttft = None
    stream = await client.chat.completions.create(
        model=MODEL,
        stream=True,
        messages=[{"role": "user", "content": message}],
        stream_options={"include_usage": True},
    )
    async for chunk in stream:
        usage = chunk.usage
        if usage:
            yield f"event: usage\ndata: {json.dumps(usage.model_dump())}\n\n"
            continue
        if chunk.choices:
            text = chunk.choices[0].delta.content or ""
            if text:
                chunk_count = chunk_count + 1
                if chunk_count == 1:
                    ttft_endtime = perf_counter()
                    ttft = ttft_endtime - request_starttime
                yield f"data: {json.dumps({'text': text})}\n\n"
    yield measure_and_done(request_starttime, ttft, chunk_count)


@app.post("/chat")
async def chat(body: LogIn, client: ClientDep):
    request_starttime = perf_counter()
    return StreamingResponse(events(client, body.log, request_starttime), media_type="text/event-stream")
