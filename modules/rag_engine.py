import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()


def chunk_text(text, chunk_size=384, overlap=64):
    """Split text into overlapping chunks."""
    if not text:
        return []
    words = text.split()
    if len(words) == 0:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def build_vector_index(documents):
    """Build sparse retrieval data (TF-IDF only, no sentence-transformers)."""
    all_chunks = []

    for doc in documents:
        if not doc.get("text"):
            continue
        chunks = chunk_text(doc["text"])

        for chunk in chunks:
            all_chunks.append({
                "document_name": doc.get("document_name", "Unknown"),
                "text": chunk
            })

    if len(all_chunks) == 0:
        return None

    texts = [chunk["text"] for chunk in all_chunks]

    # Use TF-IDF only (no sentence-transformers to avoid import errors)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    sparse_matrix = vectorizer.fit_transform(texts)

    index_data = {
        "chunks": all_chunks,
        "vectorizer": vectorizer,
        "sparse_matrix": sparse_matrix,
    }

    return index_data


def call_llm_for_answer(question, context_chunks, top_k=3):
    """Call LLM to generate answer from retrieved chunks."""
    context = "\n\n".join([chunk["text"] for chunk in context_chunks[:top_k]])
    
    if len(context) > 8000:
        context = context[:8000] + "..."

    prompt = f"""You are a logistics document assistant. Answer the question based ONLY on the provided document context.

CONTEXT:
{context}

QUESTION: {question}

ANSWER DIRECTLY AND CONCISELY. If the answer is not in the context, say "The document does not contain this information."

ANSWER:"""

    # Try Gemini (free)
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini error: {e}")

    # Try OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
            request = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                response_data = json.loads(response.read().decode("utf-8"))
            return response_data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"OpenAI error: {e}")

    # Fallback: return formatted context
    return f"""Based on the document:

{context[:800]}

(Add GOOGLE_API_KEY to .env file for AI-generated answers)"""


def answer_question(question, index_data, top_k=3):
    """Retrieve relevant chunks and generate answer using LLM."""
    if index_data is None:
        return "No indexed document content available. Please build an index first.", []

    # Retrieve using TF-IDF
    sparse_query = index_data["vectorizer"].transform([question])
    sparse_scores = cosine_similarity(sparse_query, index_data["sparse_matrix"]).flatten()

    indices = np.argsort(sparse_scores)[::-1][:top_k]

    retrieved_sources = []
    for idx in indices:
        if idx != -1 and idx < len(index_data["chunks"]):
            source = dict(index_data["chunks"][idx])
            source["retrieval_score"] = round(float(sparse_scores[idx]), 4)
            retrieved_sources.append(source)

    if len(retrieved_sources) == 0:
        return "I could not find relevant information from the processed documents.", []

    # Generate answer using LLM
    answer = call_llm_for_answer(question, retrieved_sources, top_k)

    return answer, retrieved_sources