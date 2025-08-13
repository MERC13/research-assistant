import os, time, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Pinecone client
import pinecone
PINE_API_KEY = os.getenv("PINECONE_API_KEY")
PINE_ENV = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "research-chunks")

pinecone.init(api_key=PINE_API_KEY, environment=PINE_ENV)
if INDEX_NAME not in pinecone.list_indexes():
    # create index with a typical dimension for OpenAI embeddings (1536 for some models; check model dims)
    pinecone.create_index(INDEX_NAME, dimension=1536, metric="cosine")
index = pinecone.Index(INDEX_NAME)

CHUNKS_DIR = Path(os.getenv("CHUNKS_DIR", "data/chunks"))

def embed_text(text):
    # batch if needed; here we do 1 per call for clarity
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)
    return resp["data"][0]["embedding"]

def load_chunks_from_txts():
    # expects chunk JSON lines or generated from chunker
    recs = []
    for p in CHUNKS_DIR.glob("*.txt"):
        base = p.stem
        # naive chunking on the fly if there are no chunk json files:
        text = p.read_text(encoding="utf-8")
        recs.append({"id": base + "_chunk_000", "text": text, "source_file": str(p), "chunk_idx": 0})
    # alternatively, load prebuilt JSON records if you saved them.
    return recs

def upsert_chunks(recs, batch_size=10):
    vectors = []
    for i, r in enumerate(recs):
        emb = embed_text(r["text"])
        meta = {
            "source_file": r.get("source_file"),
            "chunk_idx": r.get("chunk_idx"),
            "source_id": r.get("id"),
        }
        vectors.append((r["id"], emb, meta))
        if len(vectors) >= batch_size:
            index.upsert(vectors)
            print("upserted batch", i)
            vectors = []
            time.sleep(0.5)
    if vectors:
        index.upsert(vectors)
        print("upserted final batch")

if __name__ == "__main__":
    recs = load_chunks_from_txts()
    upsert_chunks(recs)
    print("Done.")
