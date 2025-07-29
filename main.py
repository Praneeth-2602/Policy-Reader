
# main.py
import uvicorn
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(
    title="LLM Policy Information System",
    description="System to process natural language queries and retrieve relevant information from unstructured documents.",
    version="0.1.0"
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)