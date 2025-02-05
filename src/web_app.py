from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import json
from typing import AsyncGenerator, Callable
from asyncio import Queue
import asyncio

from src.data_model import LLMResponse, llmFn, LLMRequest

def create_web_app(llm_function: llmFn):
    app = FastAPI(
        title="Investor Agent", 
        description="A FastAPI server exposing the LLM workflow", 
        version="1.0.0"
    )

    class QueryRequest(BaseModel):
        query: str

    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    @app.post("/query")
    async def query(request: QueryRequest):
        async def generate_chunks() -> AsyncGenerator[str, None]:
            chunk_queue = Queue()
            
            def chunk_handler(chunk: str):            
                return chunk_queue.put_nowait(
                    f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                )
            
            llm_request = LLMRequest(
                query=request.query,
                prompt="default prompt",
                as_json=False
            )
            
            workflow_task = asyncio.create_task(
                llm_function(llm_request, chunk_handler)
            )
            
            try:
                while not workflow_task.done() or not chunk_queue.empty():
                    try:
                        chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.05)
                        yield chunk
                        
                        while not chunk_queue.empty():
                            yield await chunk_queue.get_nowait()
                            
                    except asyncio.TimeoutError:
                        if workflow_task.done() and chunk_queue.empty():
                            break
                        continue
                
                if not workflow_task.cancelled():
                    llm_response: LLMResponse = await workflow_task
                    yield f"data: {json.dumps({'type': 'complete', 'content': llm_response.model_dump()})}\n\n"
                        
            except Exception as e:
                print(f"Error in generate_chunks: {str(e)}, type: {type(e)}")
                workflow_task.cancel()
                raise
        
        return StreamingResponse(
            generate_chunks().__aiter__(),
            media_type="text/event-stream"
        )

    return app 