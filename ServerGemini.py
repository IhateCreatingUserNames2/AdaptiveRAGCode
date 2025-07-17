from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
import json
from datetime import datetime
import hashlib
from collections import defaultdict
import os
import chromadb
from chromadb.utils import embedding_functions
import math

app = Flask(__name__)
CORS(app)  # Allow browser extensions to connect

# --- CHROMA DB SETUP ---
# This sets up a persistent client that saves data to a 'chroma_db' directory
chroma_client = chromadb.PersistentClient(path="chroma_db_gte")

# Use a sentence-transformer model directly within Chroma
# This simplifies things as we don't need to call embedding_model.encode() ourselves
print("Loading GTE-LARGE embedding model for server...")
gte_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="thenlper/gte-large")

collection = chroma_client.get_or_create_collection(
    name="adaptive_rag_patterns_gte", #<-- POINT TO THE NEW COLLECTION
    embedding_function=gte_ef
)
# --- END OF CHANGE ---
print(f"ChromaDB collection '{collection.name}' loaded/created with {collection.count()} items.")
#...


# Initialize database for metadata and statistics
def init_db():
    conn = sqlite3.connect('adaptive_rag.db', timeout=10
                           )
    c = conn.cursor()

    # Patterns table for stats (vector is now in ChromaDB)
    c.execute('''CREATE TABLE IF NOT EXISTS patterns
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  pattern_hash TEXT UNIQUE,
                  query_type TEXT,
                  language TEXT,
                  pattern_text TEXT,
                  solution_approach TEXT,
                  success_count INTEGER DEFAULT 0,
                  failure_count INTEGER DEFAULT 0,
                  avg_success_rate REAL DEFAULT 0.0,
                  created_at TIMESTAMP,
                  updated_at TIMESTAMP)''')  # <-- solution_vector BLOB removed

    # Usage logs for learning
    c.execute('''CREATE TABLE IF NOT EXISTS usage_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  pattern_id INTEGER,
                  query TEXT,
                  tool TEXT,
                  success BOOLEAN,
                  user_hash TEXT,
                  created_at TIMESTAMP,
                  FOREIGN KEY(pattern_id) REFERENCES patterns(id))''')

    conn.commit()
    conn.close()


init_db()


# Helper functions
# In ServerGemini.py, replace the entire function

def extract_pattern_features(query: str) -> dict:
    """Smarter feature extractor for live queries."""
    query_lower = query.lower()

    # Query Type Detection (can remain the same)
    query_type = 'general'
    if any(word in query_lower for word in ['debug', 'error', 'fix', 'issue', 'problem', 'cannot', 'why']):
        query_type = 'debugging'
    # ... etc ...

    # --- IMPROVED LANGUAGE DETECTION ---
    languages = {
        'python': ['python', 'django', 'flask', 'pandas', 'numpy', 'matplotlib', 'pyplot', ' def '],
        'javascript': ['javascript', 'js', 'react', 'vue', 'node', 'typescript', 'ts', 'const ', 'let ', 'function '],
        'java': ['java', 'spring', 'maven', 'public class'],
        'c++': ['c++', '#include <iostream>', 'std::'],
        'c': ['c', '#include <stdio.h>', 'struct '],
        'rust': ['rust', ' fn ', 'let mut'],
        'go': ['go', 'golang', ' func '],
        'kotlin': ['kotlin', ' fun '],
        'sql': ['sql', 'query', 'database', 'postgres', 'mysql', 'select ', 'from ', 'where '],
    }

    detected_lang = 'general'
    # Check for specific language keywords first
    for lang, keywords in languages.items():
        if any(kw in query_lower for kw in keywords):
            detected_lang = lang
            break

    # --- END OF IMPROVEMENT ---

    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'how',
                  'do', 'i'}
    important_words = [w for w in query_lower.split() if w not in stop_words and len(w) > 2]
    important_words.sort()
    pattern_text = ' '.join(important_words[:5])
    pattern_hash = hashlib.md5(f"{query_type}:{detected_lang}:{pattern_text}".encode()).hexdigest()[:16]

    return {
        'pattern_hash': pattern_hash,
        'query_type': query_type,
        'language': detected_lang,
        'pattern_text': pattern_text
    }


# Removed cosine_similarity and find_similar_patterns_by_vector functions


@app.route('/api/enhance', methods=['POST'])
def enhance_query():
    data = request.json
    query = data.get('query', '')
    tool = data.get('tool', 'unknown')
    user_id = data.get('user_id', 'anonymous')

    # --- FIX STARTS HERE ---

    # 1. Get features FIRST to know the language
    features = extract_pattern_features(query)
    detected_lang = features.get('language', 'general')

    # 2. Perform a SINGLE, filtered query to Chroma
    # We only apply the filter if a specific language was detected.
    query_params = {
        "query_texts": [query],
        "n_results": 5
    }
    if detected_lang != 'general':
        query_params["where"] = {"language": detected_lang}
        print(f"Performing filtered Chroma query for language: {detected_lang}")

    results = collection.query(**query_params)

    # --- FIX ENDS HERE ---

    # The 'results' dictionary contains IDs, distances, metadatas, documents, etc.
    # We'll reformat this to match our old structure for minimal frontend changes.
    similar_patterns = []
    if results and results['ids'][0]:
        for i, doc_id in enumerate(results['ids'][0]):
            meta = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            similarity_score = math.exp(-distance * distance)

            similar_patterns.append({
                'pattern': meta.get('pattern_text', ''),
                'approach': results['documents'][0][i],
                'success_rate': float(meta.get('avg_success_rate', 0.0)),
                'usage_count': int(meta.get('success_count', 0)) + int(meta.get('failure_count', 0)),
                'similarity': similarity_score
            })

    # The logic for creating a placeholder pattern in SQLite can stay
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    c = conn.cursor()
    c.execute('SELECT id, pattern_hash FROM patterns WHERE pattern_hash = ?', (features['pattern_hash'],))
    existing = c.fetchone()
    if not existing:
        c.execute(
            '''INSERT INTO patterns (pattern_hash, query_type, language, pattern_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)''',
            (features['pattern_hash'], features['query_type'], features['language'], features['pattern_text'],
             datetime.now(), datetime.now()))
        pattern_id = c.lastrowid
    else:
        pattern_id = existing[0]

    # Log usage
    user_hash = hashlib.md5(user_id.encode()).hexdigest()[:16]
    c.execute('''INSERT INTO usage_logs (pattern_id, query, tool, user_hash, created_at)
                 VALUES (?, ?, ?, ?, ?)''',
              (pattern_id, query, tool, user_hash, datetime.now()))

    conn.commit()
    conn.close()

    enhancement = {'pattern_id': pattern_id, 'similar_patterns': similar_patterns}

    # Build the response based on Chroma search results
    if similar_patterns:
        enhancement['has_similar_patterns'] = True
        best_pattern = similar_patterns[0]
        enhancement[
            'suggested_context'] = f"""Based on a similar problem (similarity: {best_pattern['similarity']:.2f}, success rate: {best_pattern['success_rate']:.1%}):
- Past successful approach: {best_pattern['approach']}

Here is the user's question, please use this context to provide a better answer:
---
{query}
"""
    else:
        enhancement['has_similar_patterns'] = False
        enhancement['suggested_context'] = query

    return jsonify(enhancement)


@app.route('/api/feedback', methods=['POST'])
def record_feedback():
    data = request.json
    pattern_id = data.get('pattern_id')
    success = data.get('success', False)
    query = data.get('query', '')
    solution_text = data.get('solution_approach', '')

    if not pattern_id:
        return jsonify({'error': 'pattern_id required'}), 400

    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    try:
        # --- Do all SQLite writes first ---
        if success:
            c.execute('''UPDATE patterns SET success_count = success_count + 1, solution_approach = ?, updated_at = ?
                         WHERE id = ?''', (solution_text[:500], datetime.now(), pattern_id))
        else:
            c.execute('''UPDATE patterns SET failure_count = failure_count + 1, updated_at = ?
                         WHERE id = ?''', (datetime.now(), pattern_id))

        c.execute('''UPDATE patterns SET avg_success_rate = CAST(success_count AS FLOAT) / (success_count + failure_count)
                     WHERE id = ? AND (success_count + failure_count) > 0''', (pattern_id,))

        c.execute('''UPDATE usage_logs SET success = ? WHERE id = (
                       SELECT id FROM usage_logs WHERE pattern_id = ? ORDER BY created_at DESC LIMIT 1)''',
                  (success, pattern_id))

        # --- CHROMA LEARNING LOGIC ---
        if success and query and solution_text:
            # Fetch the latest stats for this pattern to store in Chroma's metadata
            c.execute('SELECT * FROM patterns WHERE id = ?', (pattern_id,))
            pattern_stats = c.fetchone()

            if pattern_stats:
                doc_id = f"pattern_{pattern_id}"

                # Convert the Row object to a plain dictionary for Chroma's metadata
                metadata_dict = dict(pattern_stats)
                # Ensure all metadata values are of a type Chroma accepts (str, int, float, bool)
                for key, value in metadata_dict.items():
                    if isinstance(value, datetime):
                        metadata_dict[key] = str(value)

                # Use upsert for safety
                collection.upsert(
                    ids=[doc_id],
                    documents=[solution_text],
                    metadatas=[metadata_dict]
                )
                print(f"Upserted document '{doc_id}' in ChromaDB.")

        # --- Commit the transaction at the very end ---
        conn.commit()

    except Exception as e:
        print(f"An error occurred during feedback processing: {e}")
        conn.rollback()  # Roll back changes if anything fails
        return jsonify({'error': 'Internal server error during database operation'}), 500
    finally:
        conn.close()

    return jsonify({'status': 'recorded'})


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    c = conn.cursor()

    stats = {}

    # Total patterns
    c.execute('SELECT COUNT(*) FROM patterns')
    stats['total_patterns'] = c.fetchone()[0]

    # Patterns by type
    c.execute('''SELECT query_type, COUNT(*), AVG(avg_success_rate)
                 FROM patterns
                 GROUP BY query_type''')
    stats['by_type'] = {row[0]: {'count': row[1], 'avg_success': row[2] or 0}
                        for row in c.fetchall()}

    # Patterns by language
    c.execute('''SELECT language, COUNT(*), AVG(avg_success_rate)
                 FROM patterns
                 GROUP BY language''')
    stats['by_language'] = {row[0]: {'count': row[1], 'avg_success': row[2] or 0}
                            for row in c.fetchall()}

    # Top successful patterns from SQLite
    c.execute('''SELECT pattern_text, solution_approach, avg_success_rate, 
                        success_count + failure_count as total_uses
                 FROM patterns
                 WHERE avg_success_rate > 0.8 AND total_uses > 5
                 ORDER BY avg_success_rate DESC, total_uses DESC
                 LIMIT 10''')
    stats['top_patterns'] = [{'pattern': row[0], 'approach': row[1],
                              'success_rate': row[2], 'uses': row[3]}
                             for row in c.fetchall()]

    # Add ChromaDB stats
    stats['chroma_db_count'] = collection.count()

    # Usage over time (last 7 days)
    c.execute('''SELECT DATE(created_at) as date, COUNT(*) as count
                 FROM usage_logs
                 WHERE created_at > datetime('now', '-7 days')
                 GROUP BY DATE(created_at)
                 ORDER BY date''')
    stats['usage_timeline'] = {row[0]: row[1] for row in c.fetchall()}

    conn.close()

    return jsonify(stats)


@app.route('/')
def dashboard():
    """Simple dashboard"""
    # Unchanged from previous versions, will work as is.
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Adaptive RAG Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            h1 { color: #333; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .stat-card { background: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #ddd; }
            .stat-number { font-size: 2em; font-weight: bold; color: #007bff; }
            .pattern-list { margin-top: 20px; }
            .pattern-item { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; }
            .success-rate { color: #28a745; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Adaptive RAG System Dashboard</h1>
            <div id="stats">Loading stats...</div>
        </div>

        <script>
            async function loadStats() {
                try {
                    const response = await fetch('/api/stats');
                    const stats = await response.json();

                    let html = '<div class="stats-grid">';

                    // Total patterns
                    html += `<div class="stat-card">
                        <h3>Total SQLite Patterns</h3>
                        <div class="stat-number">${stats.total_patterns}</div>
                    </div>`;

                    // Total Chroma Vectors
                    html += `<div class="stat-card">
                        <h3>Learned Chroma Vectors</h3>
                        <div class="stat-number">${stats.chroma_db_count}</div>
                    </div>`;

                    // By type
                    html += '<div class="stat-card"><h3>Patterns by Type</h3>';
                    for (const [type, data] of Object.entries(stats.by_type)) {
                        html += `<div>${type}: ${data.count} (${(data.avg_success * 100).toFixed(1)}% success)</div>`;
                    }
                    html += '</div>';

                    // By language
                    html += '<div class="stat-card"><h3>Patterns by Language</h3>';
                    for (const [lang, data] of Object.entries(stats.by_language)) {
                        html += `<div>${lang}: ${data.count} (${(data.avg_success * 100).toFixed(1)}% success)</div>`;
                    }
                    html += '</div>';

                    html += '</div>';

                    // Top patterns
                    if (stats.top_patterns.length > 0) {
                        html += '<div class="pattern-list"><h2>🏆 Top Successful Patterns (from SQLite)</h2>';
                        for (const pattern of stats.top_patterns) {
                            html += `<div class="pattern-item">
                                <strong>${pattern.pattern}</strong><br>
                                Approach: ${pattern.approach || 'Not specified'}<br>
                                <span class="success-rate">Success Rate: ${(pattern.success_rate * 100).toFixed(1)}%</span> 
                                (${pattern.uses} uses)
                            </div>`;
                        }
                        html += '</div>';
                    }

                    document.getElementById('stats').innerHTML = html;
                } catch (error) {
                    document.getElementById('stats').innerHTML = '<p>Error loading stats: ' + error + '</p>';
                }
            }

            loadStats();
            setInterval(loadStats, 30000); // Refresh every 30 seconds
        </script>
    </body>
    </html>
    '''


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8002))
    app.run(host='0.0.0.0', port=port, debug=True)