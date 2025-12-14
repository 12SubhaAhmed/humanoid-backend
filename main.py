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
    allow_origins=["https://hackathon-2025-dc7z869zd-subha-sajjads-projects.vercel.app/"],      
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(q: Question):
    result = answer(q.query)

    return {
        "answer": result.get("answer", "I don't know")
    }



