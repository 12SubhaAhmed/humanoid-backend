import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import cohere
import google.generativeai as genai
import google.generativeai.types as gtypes

# ----------------------------
# Load API keys from .env
# ----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ----------------------------
# Configure Gemini
# ----------------------------
genai.configure(api_key=GEMINI_API_KEY)

# ----------------------------
# Initialize Cohere & Qdrant
# ----------------------------
cohere_client = cohere.Client(COHERE_API_KEY)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ----------------------------
# Embedding function
# ----------------------------
def get_embedding(text: str):
    response = cohere_client.embed(
        model="embed-english-v3.0",
        input_type="search_query",
        texts=[text]
    )
    return response.embeddings[0]

# ----------------------------
# Retrieve relevant chunks
# ----------------------------
def retrieve_chunks(query: str):
    embedding = get_embedding(query)
    result = qdrant.query_points(
        collection_name="humanoid_robotics",
        query=embedding,
        limit=5
    )
    return [p.payload["text"] for p in result.points]


def answer(question: str):
    # Step 1 — Retrieve top chunks
    retrieved = retrieve_chunks(question)
    if not retrieved:
        retrieved_text = "No relevant content found."
    else:
        retrieved_text = "\n\n".join(retrieved[:3])

    # Step 2 — Prepare prompt with real RAG context
    prompt = (
    f"You are an AI tutor for the Physical AI & Humanoid Robotics textbook.\n"
    f"Use ONLY the retrieved content below to answer.\n"
    f"Write the answer in **3–5 complete sentences**, fully explaining the concept.\n"
    f"If the answer is not in the retrieved content, reply: 'I don't know'.\n\n"
    f"### Retrieved Content:\n{retrieved_text}\n\n"
    f"### Question:\n{question}\n\n"
    f"### Answer:"
    )

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        contents=prompt,
        generation_config=gtypes.GenerationConfig(
            temperature=0.5,
            max_output_tokens=1000
        )
    )

    return {
        "answer": response.text.strip()
    }
