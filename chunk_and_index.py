import os
import pickle
import re
from sentence_transformers import SentenceTransformer
import faiss

def load_documents(folder_path):
    texts = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.txt'):
            with open(os.path.join(folder_path, fname), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
    return texts

def clean_text(text):
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9.,;:!?() ]', '', text)
    return text.strip()

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        if len(chunk.split()) > 10:  # Skip very small chunks
            chunks.append(chunk)
    return chunks

def build_faiss_index(chunks, model):
    print("🔁 Encoding chunks into vectors...")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

if __name__ == "__main__":
    folder_path = 'data/pdf_text'

    print("📂 Loading documents...")
    docs = load_documents(folder_path)

    if not docs:
        print("⚠️ No .txt files found in 'data/pdf_text'. Exiting.")
        exit()

    print("🧹 Cleaning documents...")
    cleaned_docs = [clean_text(doc) for doc in docs]

    print("✂️ Splitting into chunks...")
    all_chunks = []
    for doc in cleaned_docs:
        all_chunks.extend(chunk_text(doc))

    print(f"✅ Total Chunks Created: {len(all_chunks)}")

    print("💡 Loading embedding model...")
    model = SentenceTransformer("all-mpnet-base-v2")

    print("🔧 Building FAISS index...")
    index = build_faiss_index(all_chunks, model)

    os.makedirs('embeddings/faiss_index', exist_ok=True)
    print("💾 Saving index and chunks...")
    faiss.write_index(index, 'embeddings/faiss_index/index.faiss')

    with open('data/chunks.pkl', 'wb') as f:
        pickle.dump(all_chunks, f)

    print("🎉 Done! FAISS index and chunks saved successfully.")
