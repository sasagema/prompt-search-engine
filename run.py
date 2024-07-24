import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from promptSearchEngine import PromptSearchEngine
from vectorizer import Vectorizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DATASET = "Gustavosta/Stable-Diffusion-Prompts"


    
model = SentenceTransformer(EMBEDDING_MODEL)
dataset = load_dataset(DATASET , split="train[:1%]")
promptSearchEngine = PromptSearchEngine(dataset["Prompt"], model)

class SearchRequest(BaseModel):
    query: str 
    n: int | None = 5

app = FastAPI()

@app.get("/")
async def root():
    return {"message": 'GET /docs'}

@app.get("/search")
async def search(q: str, n: int = 5):
    results = []
    if q.isspace() or q =="":
        return {"message": "Enter query"}
    else:
        results = promptSearchEngine.most_similar(q, n)
    if not results:
        raise HTTPException(status_code=404, detail="No prompts found.")
    return promptSearchEngine.stringify_prompts(results)


@app.post("/search")
async def searchPost(request: SearchRequest):
    results = promptSearchEngine.most_similar(request.query, request.n)
    if not results:
        raise HTTPException(status_code=404, detail="No prompts found.")
    formatted_results = [{"similarity": float(similarity), "prompt": prompt } for similarity, prompt in results]
    return { "data" : formatted_results }
