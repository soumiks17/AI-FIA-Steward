import json
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def build_vector_database(jsonl_path, persist_dir):
    documents = []
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
        
    for line in tqdm(lines, desc="Parsing JSONL"):
        if not line.strip():
            continue
        
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        content = data.get("reasoning", "")
        if content == "N/A" or not content:
            continue
            
        metadata = {
            "driver": str(data.get("driver", "Unknown")),
            "breach": str(data.get("breach", "Unknown")),
            "decision": str(data.get("decision", "Unknown")),
            "source": str(data.get("source_file", "Unknown"))
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
        
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    batch_size = 100
    for i in tqdm(range(0, len(documents), batch_size), desc="Embedding Documents"):
        batch = documents[i:i + batch_size]
        db.add_documents(batch)
        
    return db

if __name__ == "__main__":
    build_vector_database("structured_precedents.jsonl", "./fia_chroma_db")