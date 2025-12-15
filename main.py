from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from agent import answer

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Humanoid Backend is running 🚀"}
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
        "http://localhost:5173",
        "https://hackathon-2025-6nha2r6gr-subha-sajjads-projects.vercel.app"],      
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(q: Question):
    result = answer(q.query)

    return {
        "answer": result.get("answer") or "Let me think about that and help you."
    }







