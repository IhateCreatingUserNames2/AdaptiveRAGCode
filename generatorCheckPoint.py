# generator2.py
# FINAL-RESUMABLE VERSION - For full dataset ingestion with checkpointing.

import os
import sqlite3
from datetime import datetime
import hashlib
import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

print("--- Adaptive RAG System Bulk Ingestion (Resumable) ---")

# --- Configuration ---
CHECKPOINT_FILE = 'ingestion_checkpoint.txt'
SQLITE_DB_PATH = 'adaptive_rag.db'
CHROMA_DB_PATH = "chroma_db"

# --- Database & Model Setup ---
print("1. Initializing database connections...")
try:
    chroma_client = chromadb.PersistentClient(path="chroma_db_gte")  # <-- USE A NEW FOLDER

    # --- THIS IS THE CHANGE ---
    print("Loading GTE-LARGE embedding model...")
    gte_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-large")

    collection = chroma_client.get_or_create_collection(
        name="adaptive_rag_patterns_gte",  # <-- USE A NEW COLLECTION NAME
        embedding_function=gte_ef
    )
    # --- END OF CHANGE ---

    SQLITE_DB_PATH = 'adaptive_rag.db'  # You can reuse the SQLite DB
    print(f"   - ChromaDB connection successful. Collection has {collection.count()} items.")
    # ...
except Exception as e:
    print(f"🔥 FATAL ERROR: Could not connect to ChromaDB. Error: {e}")
    exit()


# --- Checkpoint Functions ---
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_checkpoint(processed_count):
    with open(CHECKPOINT_FILE, 'w') as f:
        f.write(str(processed_count))


# --- Helper Functions (Same as before) ---
def extract_features_for_ingestion(query: str, code: str) -> dict:
    # ... (This function remains unchanged from the previous version)
    # --- Language Detection (Primary Change) ---
    detected_lang = 'general'
    lang_keywords_in_code = {
        'python': ['def ', 'import ', 'class ', 'elif ', 'self.'],
        'javascript': ['function ', 'const ', 'let ', 'import ', '=>', 'async '],
        'java': ['public class', 'import java.util', 'System.out.println', 'public static void'],
        'c++': ['#include <iostream>', 'std::', 'int main()', '->', '::'],
        'c': ['#include <stdio.h>', 'int main(void)', 'printf(', 'struct '],
        'rust': ['fn ', 'let mut', '::<>', 'println!'],
        'go': ['package main', 'import (', 'func ', ':=', 'fmt.Println'],
        'kotlin': ['fun ', 'val ', 'var ', 'import kotlin']
    }

    # Check for language keywords in the code first for higher accuracy
    for lang, keywords in lang_keywords_in_code.items():
        if any(kw in code for kw in keywords):
            detected_lang = lang
            break

    # If code detection fails, fall back to checking the query text
    if detected_lang == 'general':
        query_lower_for_lang = query.lower()
        lang_keywords_in_query = {
            'python': ['python'], 'javascript': ['javascript', 'js', 'typescript', 'ts'],
            'java': ['java'], 'c++': ['c++'], 'c': ['c'], 'rust': ['rust'],
            'go': ['go'], 'kotlin': ['kotlin']
        }
        for lang, keywords in lang_keywords_in_query.items():
            if any(kw in query_lower_for_lang for kw in keywords):
                detected_lang = lang
                break

    # --- Query Type and Pattern Text (can remain similar) ---
    query_lower = query.lower()
    query_type = 'general'
    if any(word in query_lower for word in ['debug', 'error', 'fix', 'issue', 'problem', 'cannot', 'why']):
        query_type = 'debugging'
    elif any(word in query_lower for word in ['optimize', 'performance', 'faster', 'improve', 'slow']):
        query_type = 'optimization'
    elif any(word in query_lower for word in
             ['create', 'build', 'implement', 'make', 'how to', 'how do i', 'write a function']):
        query_type = 'creation'
    elif any(word in query_lower for word in ['refactor', 'clean', 'reorganize', 'best way', 'rewrite']):
        query_type = 'refactoring'

    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'how',
                  'do', 'i', 'given', 'as', 'per', 'following'}
    important_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    important_words.sort()
    pattern_text = ' '.join(important_words[:5])
    pattern_hash = hashlib.md5(f"{query_type}:{detected_lang}:{pattern_text}".encode()).hexdigest()[:16]

    return {'query_type': query_type, 'language': detected_lang, 'pattern_text': pattern_text,
            'pattern_hash': pattern_hash}


# --- Main Ingestion Logic ---
def process_batch(df_batch: pd.DataFrame):
    # ... (This function remains unchanged from the previous version)
    unique_patterns = {}

    for _, row in df_batch.iterrows():
        query, code_completion = row.get('prompt'), row.get('completion')
        if not isinstance(query, str) or not query.strip() or not isinstance(code_completion,
                                                                             str) or not code_completion.strip():
            continue

        features = extract_features_for_ingestion(query, code_completion)
        pattern_hash = features['pattern_hash']

        if pattern_hash not in unique_patterns:
            unique_patterns[pattern_hash] = {
                'features': features,
                'original_query': query,
                'solution_text': f"Based on the prompt, the following code provides the solution:\n\n```\n{code_completion}\n```"
            }

    if not unique_patterns: return 0

    sqlite_patterns_to_insert = [
        (data['features']['pattern_hash'], data['features']['query_type'], data['features']['language'],
         data['features']['pattern_text'], data['solution_text'][:500],
         1, 0, 1.0, datetime.now(), datetime.now())
        for data in unique_patterns.values()
    ]

    with sqlite3.connect(SQLITE_DB_PATH, timeout=30) as conn:
        c = conn.cursor()
        c.executemany(
            'INSERT OR IGNORE INTO patterns (pattern_hash, query_type, language, pattern_text, solution_approach, success_count, failure_count, avg_success_rate, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            sqlite_patterns_to_insert)
        conn.commit()

    chroma_docs = {'ids': [], 'documents': [], 'metadatas': []}
    with sqlite3.connect(SQLITE_DB_PATH, timeout=30) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        hashes_to_query = tuple(unique_patterns.keys())
        c.execute(f"SELECT * FROM patterns WHERE pattern_hash IN ({','.join(['?'] * len(hashes_to_query))})",
                  hashes_to_query)
        hash_to_db_row = {row['pattern_hash']: row for row in c.fetchall()}

    for pattern_hash, data in unique_patterns.items():
        db_row = hash_to_db_row.get(pattern_hash)
        if db_row:
            pattern_id = db_row['id']
            metadata_dict = dict(db_row)
            metadata_dict['original_query'] = data['original_query']
            for key, value in metadata_dict.items():
                if isinstance(value, datetime): metadata_dict[key] = str(value)

            chroma_docs['ids'].append(f"pattern_{pattern_id}")
            chroma_docs['documents'].append(data['solution_text'])
            chroma_docs['metadatas'].append(metadata_dict)

    if chroma_docs['ids']:
        collection.upsert(ids=chroma_docs['ids'], documents=chroma_docs['documents'],
                          metadatas=chroma_docs['metadatas'])

    return len(chroma_docs['ids'])


def run_ingestion(batch_size=1000, max_items=None):
    """Loads, shuffles, and processes the dataset with checkpointing."""

    start_from = load_checkpoint()

    print("\n2. Loading microsoft/NextCoderDataset...")
    try:
        dataset = load_dataset("microsoft/NextCoderDataset", split='train')
    except Exception as e:
        print(f"🔥 FATAL ERROR: Failed to load dataset. Error: {e}")
        return

    print("🔀 Shuffling dataset with a fixed seed for consistency...")
    # Using a fixed seed is VITAL for checkpointing to work.
    # It ensures the order of items is the same every time you run the script.
    shuffled_dataset = dataset.shuffle(seed=42)

    dataset_size = len(shuffled_dataset)
    if max_items:
        dataset_size = min(dataset_size, max_items)

    print(f"✅ Dataset ready. Total items: {len(shuffled_dataset)}.")
    if start_from > 0:
        print(f"🔄 Resuming from checkpoint. Skipping first {start_from} items.")

    print("\n3. Starting ingestion process...")
    total_items_in_db = collection.count()
    batch, items_processed_this_run = [], 0

    # Set up the progress bar
    pbar = tqdm(total=dataset_size, initial=start_from, desc="Processing Items", unit="item")

    # Use .select() to create a view of the dataset starting from the checkpoint
    resumable_dataset = shuffled_dataset.select(range(start_from, dataset_size))

    for item in resumable_dataset:
        batch.append(item)

        if len(batch) >= batch_size:
            process_batch(pd.DataFrame(batch))
            items_processed_this_run += len(batch)
            pbar.update(len(batch))
            save_checkpoint(start_from + items_processed_this_run)
            batch = []

    if batch:
        process_batch(pd.DataFrame(batch))
        items_processed_this_run += len(batch)
        pbar.update(len(batch))
        save_checkpoint(start_from + items_processed_this_run)

    pbar.close()
    print("\n--- Ingestion Complete! ---")
    final_count = collection.count()
    print(f"✅ Items processed in this run: {items_processed_this_run}")
    print(f"📈 New vectors added to ChromaDB: {final_count - total_items_in_db}")
    print(f"📊 ChromaDB now contains a total of {final_count} vectors.")


if __name__ == "__main__":
    run_ingestion(batch_size=1000, max_items=None)