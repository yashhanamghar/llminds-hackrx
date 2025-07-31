import faiss
import pickle
from sentence_transformers import SentenceTransformer
from google.generativeai import configure, GenerativeModel
import json

# Configure Gemini
configure(api_key="AIzaSyAUL_7VDhdMgcurGynHx-b5ZothJmjzGo8")
model_gemini = GenerativeModel("models/gemini-1.5-flash")

# Load FAISS and chunks
faiss_index = faiss.read_index("embeddings/faiss_index/index.faiss")
with open("data/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

embedding_model = SentenceTransformer("all-mpnet-base-v2")

def get_top_chunks(query, top_k=3):  # Reduced for speed
    query_embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def get_answer(query):
    relevant_chunks = get_top_chunks(query)
    context = "\n".join(relevant_chunks)

    prompt = f"""
    You are a helpful AI assistant.

    Use the context below to answer the question in 2 lines or less.
    Respond ONLY with a valid JSON object in this exact format:

    {{
      "answer": "short precise factual answer here"
    }}

    Context:
    {context}

    Question: {query}
    """

    try:
        response = model_gemini.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        raw_text = response.text.strip()

        if raw_text.startswith("```"):
            raw_text = raw_text.strip("```").replace("json", "").strip()

        json_output = json.loads(raw_text)

        # Enforce structure and remove extras
        return {
            "answer": json_output.get("answer", "")[:300]  # 2-line limit (approx)
        }

    except Exception as e:
        return {
            "answer": f"Error generating response: {str(e)}"
        }
