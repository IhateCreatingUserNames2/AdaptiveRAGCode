#!/usr/bin/env python3
"""
Enhanced Adaptive RAG MCP Server with Knowledge Ingestion

This enhanced version includes tools for users to feed their own knowledge
into the system, making it truly adaptive to their specific codebase and practices.
"""

import os
import sqlite3
from datetime import datetime
import hashlib
import chromadb
from chromadb.utils import embedding_functions
import math
from typing import Dict, Any, List, Optional, Union
import json
import ast
from pathlib import Path

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field


# --- Initialize system (same as before) ---
def init_rag_system():
    """Initialize the RAG system components"""
    chroma_client = chromadb.PersistentClient(path="chroma_db_gte")
    gte_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="thenlper/gte-large"
    )
    collection = chroma_client.get_or_create_collection(
        name="adaptive_rag_patterns_gte",
        embedding_function=gte_ef
    )
    
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    c = conn.cursor()
    
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
                  source TEXT DEFAULT 'interactive',
                  created_at TIMESTAMP,
                  updated_at TIMESTAMP)''')
    
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
    
    return collection

collection = init_rag_system()
mcp = FastMCP("AdaptiveRAG")

# --- Helper functions ---
def extract_pattern_features(query: str) -> Dict[str, str]:
    """Extract key features from a coding query for pattern matching"""
    query_lower = query.lower()
    
    query_type = 'general'
    if any(word in query_lower for word in ['debug', 'error', 'fix', 'issue', 'problem', 'cannot', 'why']):
        query_type = 'debugging'
    elif any(word in query_lower for word in ['optimize', 'performance', 'faster', 'improve', 'slow']):
        query_type = 'optimization'
    elif any(word in query_lower for word in ['create', 'build', 'implement', 'make', 'how to', 'write']):
        query_type = 'creation'
    elif any(word in query_lower for word in ['refactor', 'clean', 'reorganize', 'best way']):
        query_type = 'refactoring'
    
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
    for lang, keywords in languages.items():
        if any(kw in query_lower for kw in keywords):
            detected_lang = lang
            break
    
    stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'how', 'do', 'i'}
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

def store_custom_pattern(query: str, solution: str, language: str, source: str, success_rate: float = 1.0):
    """Store a custom pattern in the knowledge base"""
    features = extract_pattern_features(query)
    pattern_hash = hashlib.md5(f"custom:{language}:{query}".encode()).hexdigest()[:16]
    
    # Store in SQLite
    with sqlite3.connect('adaptive_rag.db', timeout=10) as conn:
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO patterns 
                     (pattern_hash, query_type, language, pattern_text, solution_approach, 
                      success_count, failure_count, avg_success_rate, source, created_at, updated_at) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (pattern_hash, features['query_type'], language, features['pattern_text'], 
                   solution[:500], 1, 0, success_rate, source, datetime.now(), datetime.now()))
        
        # Get pattern ID
        c.execute('SELECT id FROM patterns WHERE pattern_hash = ?', (pattern_hash,))
        result = c.fetchone()
        pattern_id = result[0] if result else c.lastrowid
    
    # Store in ChromaDB
    doc_id = f"custom_pattern_{pattern_id}"
    metadata = {
        "id": pattern_id,
        "pattern_hash": pattern_hash,
        "query_type": features['query_type'],
        "language": language,
        "pattern_text": features['pattern_text'],
        "avg_success_rate": success_rate,
        "success_count": 1,
        "failure_count": 0,
        "source": source,
        "created_at": str(datetime.now()),
        "updated_at": str(datetime.now())
    }
    
    collection.upsert(
        ids=[doc_id],
        documents=[solution],
        metadatas=[metadata]
    )
    
    return pattern_id

# --- Core RAG Tools (from previous version) ---
class EnhancedContext(BaseModel):
    query: str = Field(description="The original query")
    has_similar_patterns: bool = Field(description="Whether similar patterns were found")
    suggested_context: str = Field(description="Enhanced context to provide to the LLM")
    pattern_id: int = Field(description="Internal pattern ID for feedback")
    similar_patterns: List[Dict[str, Any]] = Field(description="List of similar successful patterns")

@mcp.tool()
def get_enhanced_context(query: str, tool: str = "mcp_client", user_id: str = "anonymous") -> EnhancedContext:
    """Get enhanced context for a coding query by finding similar successful solutions."""
    features = extract_pattern_features(query)
    detected_lang = features.get('language', 'general')
    
    query_params = {"query_texts": [query], "n_results": 5}
    if detected_lang != 'general':
        query_params["where"] = {"language": detected_lang}
    
    results = collection.query(**query_params)
    
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
                'similarity': similarity_score,
                'source': meta.get('source', 'unknown')
            })
    
    # Create or find pattern in SQLite
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    c = conn.cursor()
    c.execute('SELECT id FROM patterns WHERE pattern_hash = ?', (features['pattern_hash'],))
    existing = c.fetchone()
    
    if not existing:
        c.execute(
            '''INSERT INTO patterns (pattern_hash, query_type, language, pattern_text, created_at, updated_at) 
               VALUES (?, ?, ?, ?, ?, ?)''',
            (features['pattern_hash'], features['query_type'], features['language'], 
             features['pattern_text'], datetime.now(), datetime.now()))
        pattern_id = c.lastrowid
    else:
        pattern_id = existing[0]
    
    user_hash = hashlib.md5(user_id.encode()).hexdigest()[:16]
    c.execute('''INSERT INTO usage_logs (pattern_id, query, tool, user_hash, created_at)
                 VALUES (?, ?, ?, ?, ?)''',
              (pattern_id, query, tool, user_hash, datetime.now()))
    
    conn.commit()
    conn.close()
    
    # Build enhanced context
    if similar_patterns:
        best_pattern = similar_patterns[0]
        context_parts = [
            f"Based on a similar problem (similarity: {best_pattern['similarity']:.2f}, success rate: {best_pattern['success_rate']:.1%}):",
            f"- Past successful approach: {best_pattern['approach']}"
        ]
        
        if best_pattern['source'] != 'interactive':
            context_parts.append(f"- Source: {best_pattern['source']}")
        
        if len(similar_patterns) > 1:
            context_parts.append(f"\nAdditional context from {len(similar_patterns)-1} similar cases:")
            for p in similar_patterns[1:3]:
                snippet = p['approach'][:100] + "..." if len(p['approach']) > 100 else p['approach']
                context_parts.append(f"- {snippet} (Source: {p['source']})")
        
        context_parts.append(f"\nHere is the user's question, please use this context to provide a better answer:\n---\n{query}")
        suggested_context = "\n".join(context_parts)
    else:
        suggested_context = f"No similar patterns found in knowledge base. Treating as new problem:\n---\n{query}"
    
    return EnhancedContext(
        query=query,
        has_similar_patterns=len(similar_patterns) > 0,
        suggested_context=suggested_context,
        pattern_id=pattern_id,
        similar_patterns=similar_patterns
    )

class FeedbackResult(BaseModel):
    status: str = Field(description="Status of the feedback recording")
    learned: bool = Field(description="Whether the system learned from this feedback")
    pattern_id: int = Field(description="Pattern ID that was updated")

@mcp.tool()
def record_solution_feedback(pattern_id: int, query: str, solution_approach: str, success: bool) -> FeedbackResult:
    """Record feedback about whether a solution worked or not."""
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    try:
        if success:
            c.execute('''UPDATE patterns SET success_count = success_count + 1, 
                        solution_approach = ?, updated_at = ? WHERE id = ?''', 
                     (solution_approach[:500], datetime.now(), pattern_id))
        else:
            c.execute('''UPDATE patterns SET failure_count = failure_count + 1, 
                        updated_at = ? WHERE id = ?''', 
                     (datetime.now(), pattern_id))
        
        c.execute('''UPDATE patterns SET avg_success_rate = 
                    CAST(success_count AS FLOAT) / (success_count + failure_count)
                    WHERE id = ? AND (success_count + failure_count) > 0''', (pattern_id,))
        
        c.execute('''UPDATE usage_logs SET success = ? WHERE id = (
                       SELECT id FROM usage_logs WHERE pattern_id = ? 
                       ORDER BY created_at DESC LIMIT 1)''',
                  (success, pattern_id))
        
        learned = False
        if success and query and solution_approach:
            c.execute('SELECT * FROM patterns WHERE id = ?', (pattern_id,))
            pattern_stats = c.fetchone()
            
            if pattern_stats:
                doc_id = f"pattern_{pattern_id}"
                metadata_dict = dict(pattern_stats)
                
                for key, value in metadata_dict.items():
                    if isinstance(value, datetime):
                        metadata_dict[key] = str(value)
                
                collection.upsert(
                    ids=[doc_id],
                    documents=[solution_approach],
                    metadatas=[metadata_dict]
                )
                learned = True
        
        conn.commit()
        status = "Feedback recorded successfully"
        
    except Exception as e:
        conn.rollback()
        status = f"Error recording feedback: {e}"
        learned = False
    finally:
        conn.close()
    
    return FeedbackResult(status=status, learned=learned, pattern_id=pattern_id)

# --- NEW: Knowledge Ingestion Tools ---

class CustomKnowledge(BaseModel):
    """Custom knowledge entry"""
    question: str = Field(description="The question or problem description")
    answer: str = Field(description="The solution or answer")
    language: str = Field(default="general", description="Programming language (python, javascript, etc.)")
    source: str = Field(default="manual_entry", description="Source of this knowledge")
    success_rate: float = Field(default=1.0, description="Expected success rate (0.0 to 1.0)", ge=0.0, le=1.0)

class IngestionResult(BaseModel):
    """Result of knowledge ingestion"""
    success: bool = Field(description="Whether ingestion was successful")
    patterns_added: int = Field(description="Number of patterns added")
    message: str = Field(description="Status message")
    pattern_ids: List[int] = Field(description="IDs of created patterns")

@mcp.tool()
def add_custom_knowledge(knowledge_entries: List[CustomKnowledge]) -> IngestionResult:
    """
    Add custom knowledge entries to the RAG system.
    
    Use this to teach the system about your specific codebase, company practices,
    or any solutions that worked well for your team.
    """
    pattern_ids = []
    errors = []
    
    for entry in knowledge_entries:
        try:
            pattern_id = store_custom_pattern(
                query=entry.question,
                solution=entry.answer,
                language=entry.language,
                source=entry.source,
                success_rate=entry.success_rate
            )
            pattern_ids.append(pattern_id)
        except Exception as e:
            errors.append(f"Error adding '{entry.question[:50]}...': {e}")
    
    success = len(pattern_ids) > 0
    patterns_added = len(pattern_ids)
    
    if errors:
        message = f"Added {patterns_added} patterns. Errors: {'; '.join(errors)}"
    else:
        message = f"Successfully added {patterns_added} knowledge patterns"
    
    return IngestionResult(
        success=success,
        patterns_added=patterns_added,
        message=message,
        pattern_ids=pattern_ids
    )

@mcp.tool()
def extract_knowledge_from_code(code_content: str, file_path: str = "unknown", language: str = "python") -> IngestionResult:
    """
    Extract knowledge patterns from code content.
    
    This tool analyzes code and extracts function definitions, docstrings,
    and comments to create knowledge patterns automatically.
    """
    pattern_ids = []
    errors = []
    
    try:
        if language == "python":
            # Parse Python code
            tree = ast.parse(code_content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_name = node.name
                    docstring = ast.get_docstring(node)
                    
                    if docstring and len(docstring) > 20:
                        try:
                            # Get function source
                            func_source = ast.get_source_segment(code_content, node)
                            if not func_source:
                                # Fallback: extract function manually
                                lines = code_content.split('\n')
                                start_line = node.lineno - 1
                                # Find the end of the function
                                end_line = start_line + 1
                                base_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                                
                                for i in range(start_line + 1, len(lines)):
                                    line = lines[i]
                                    if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
                                        break
                                    end_line = i + 1
                                
                                func_source = '\n'.join(lines[start_line:end_line])
                            
                            # Create synthetic question from function name and docstring
                            question = f"How to implement {func_name}"
                            if docstring:
                                question += f": {docstring[:100]}"
                            
                            pattern_id = store_custom_pattern(
                                query=question,
                                solution=func_source,
                                language=language,
                                source=f"Code extraction: {file_path}",
                                success_rate=0.8  # Moderate confidence for extracted code
                            )
                            pattern_ids.append(pattern_id)
                            
                        except Exception as e:
                            errors.append(f"Error extracting function {func_name}: {e}")
                
                # Extract classes with docstrings
                elif isinstance(node, ast.ClassDef):
                    class_name = node.name
                    docstring = ast.get_docstring(node)
                    
                    if docstring and len(docstring) > 20:
                        try:
                            # Get class source (simplified)
                            lines = code_content.split('\n')
                            start_line = node.lineno - 1
                            class_source = f"class {class_name}:\n    {docstring}\n    # ... implementation"
                            
                            question = f"How to implement {class_name} class: {docstring[:100]}"
                            
                            pattern_id = store_custom_pattern(
                                query=question,
                                solution=class_source,
                                language=language,
                                source=f"Code extraction: {file_path}",
                                success_rate=0.7
                            )
                            pattern_ids.append(pattern_id)
                            
                        except Exception as e:
                            errors.append(f"Error extracting class {class_name}: {e}")
        
        else:
            # For non-Python code, extract comments and basic patterns
            lines = code_content.split('\n')
            current_comment = []
            current_code = []
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('//') or stripped.startswith('#'):
                    current_comment.append(stripped[2:].strip())
                elif stripped and current_comment:
                    current_code.append(line)
                elif not stripped and current_comment and current_code:
                    # End of a comment-code block
                    if len(current_comment) > 0 and len(current_code) > 0:
                        comment_text = ' '.join(current_comment)
                        code_text = '\n'.join(current_code)
                        
                        if len(comment_text) > 10 and len(code_text) > 10:
                            question = f"How to {comment_text}"
                            
                            pattern_id = store_custom_pattern(
                                query=question,
                                solution=code_text,
                                language=language,
                                source=f"Code extraction: {file_path}",
                                success_rate=0.6
                            )
                            pattern_ids.append(pattern_id)
                    
                    current_comment = []
                    current_code = []
                elif stripped:
                    current_code.append(line)
    
    except Exception as e:
        errors.append(f"Error parsing {language} code: {e}")
    
    success = len(pattern_ids) > 0
    patterns_added = len(pattern_ids)
    
    if errors:
        message = f"Extracted {patterns_added} patterns. Errors: {'; '.join(errors[:3])}"
    else:
        message = f"Successfully extracted {patterns_added} knowledge patterns from code"
    
    return IngestionResult(
        success=success,
        patterns_added=patterns_added,
        message=message,
        pattern_ids=pattern_ids
    )

class RepoAnalysis(BaseModel):
    """Repository analysis result"""
    total_files: int = Field(description="Total files analyzed")
    patterns_extracted: int = Field(description="Patterns successfully extracted")
    languages_found: List[str] = Field(description="Programming languages detected")
    summary: str = Field(description="Analysis summary")

@mcp.tool()
def analyze_repository(repo_path: str, include_extensions: List[str] = [".py", ".js", ".ts", ".java", ".go", ".rs"]) -> RepoAnalysis:
    """
    Analyze a code repository and extract knowledge patterns.
    
    This tool walks through a repository, identifies code files,
    and extracts reusable patterns automatically.
    """
    if not os.path.exists(repo_path):
        return RepoAnalysis(
            total_files=0,
            patterns_extracted=0,
            languages_found=[],
            summary=f"Repository path not found: {repo_path}"
        )
    
    total_files = 0
    total_patterns = 0
    languages_found = set()
    
    # Extension to language mapping
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.cpp': 'c++',
        '.c': 'c',
        '.rb': 'ruby',
        '.php': 'php'
    }
    
    try:
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'target', 'build', 'dist']]
            
            for file in files:
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in include_extensions:
                    file_path = os.path.join(root, file)
                    language = ext_to_lang.get(file_ext, 'general')
                    languages_found.add(language)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        
                        if len(content) > 100:  # Skip tiny files
                            total_files += 1
                            relative_path = os.path.relpath(file_path, repo_path)
                            
                            result = extract_knowledge_from_code(content, relative_path, language)
                            total_patterns += result.patterns_added
                            
                    except Exception as e:
                        continue  # Skip files that can't be read
    
    except Exception as e:
        return RepoAnalysis(
            total_files=0,
            patterns_extracted=0,
            languages_found=[],
            summary=f"Error analyzing repository: {e}"
        )
    
    summary = f"Analyzed {total_files} files and extracted {total_patterns} knowledge patterns. Found languages: {', '.join(sorted(languages_found))}"
    
    return RepoAnalysis(
        total_files=total_files,
        patterns_extracted=total_patterns,
        languages_found=list(languages_found),
        summary=summary
    )

# --- Statistics and System Info (Enhanced) ---
class SystemStats(BaseModel):
    total_patterns: int = Field(description="Total patterns in SQLite database")
    learned_vectors: int = Field(description="Total learned vectors in ChromaDB")
    patterns_by_language: Dict[str, Dict[str, Any]] = Field(description="Patterns grouped by programming language")
    patterns_by_type: Dict[str, Dict[str, Any]] = Field(description="Patterns grouped by query type")
    patterns_by_source: Dict[str, int] = Field(description="Patterns grouped by source")
    top_successful_patterns: List[Dict[str, Any]] = Field(description="Most successful patterns")

@mcp.tool()
def get_system_statistics() -> SystemStats:
    """Get comprehensive statistics about the adaptive RAG system's learning progress."""
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    c = conn.cursor()
    
    # Total patterns
    c.execute('SELECT COUNT(*) FROM patterns')
    total_patterns = c.fetchone()[0]
    
    # Patterns by language
    c.execute('''SELECT language, COUNT(*), AVG(avg_success_rate)
                 FROM patterns GROUP BY language''')
    patterns_by_language = {
        row[0]: {'count': row[1], 'avg_success': row[2] or 0}
        for row in c.fetchall()
    }
    
    # Patterns by type
    c.execute('''SELECT query_type, COUNT(*), AVG(avg_success_rate)
                 FROM patterns GROUP BY query_type''')
    patterns_by_type = {
        row[0]: {'count': row[1], 'avg_success': row[2] or 0}
        for row in c.fetchall()
    }
    
    # Patterns by source
    c.execute('''SELECT source, COUNT(*) FROM patterns GROUP BY source''')
    patterns_by_source = {row[0]: row[1] for row in c.fetchall()}
    
    # Top successful patterns
    c.execute('''SELECT pattern_text, solution_approach, avg_success_rate, 
                        success_count + failure_count as total_uses, language, query_type, source
                 FROM patterns
                 WHERE avg_success_rate > 0.7 AND total_uses > 1
                 ORDER BY avg_success_rate DESC, total_uses DESC
                 LIMIT 10''')
    top_patterns = []
    for row in c.fetchall():
        top_patterns.append({
            'pattern': row[0],
            'approach': row[1][:200] + '...' if row[1] and len(row[1]) > 200 else (row[1] or 'Not specified'),
            'success_rate': row[2],
            'uses': row[3],
            'language': row[4],
            'type': row[5],
            'source': row[6]
        })
    
    conn.close()
    
    return SystemStats(
        total_patterns=total_patterns,
        learned_vectors=collection.count(),
        patterns_by_language=patterns_by_language,
        patterns_by_type=patterns_by_type,
        patterns_by_source=patterns_by_source,
        top_successful_patterns=top_patterns
    )

# --- Resources (Enhanced) ---
@mcp.resource("adaptive-rag://dashboard")
def get_dashboard() -> str:
    """Real-time dashboard showing system learning progress"""
    stats = get_system_statistics()
    
    dashboard = f"""# Adaptive RAG System Dashboard

## Learning Progress
- **Total Patterns Tracked**: {stats.total_patterns}
- **Learned Solutions**: {stats.learned_vectors}
- **Knowledge Base Growth**: {((stats.learned_vectors / max(stats.total_patterns, 1)) * 100):.1f}% patterns have successful solutions

## Knowledge Sources
"""
    
    for source, count in sorted(stats.patterns_by_source.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / max(stats.total_patterns, 1)) * 100
        dashboard += f"- **{source}**: {count} patterns ({percentage:.1f}%)\n"
    
    dashboard += "\n## Performance by Programming Language\n"
    for lang, data in sorted(stats.patterns_by_language.items(), key=lambda x: x[1]['count'], reverse=True):
        dashboard += f"- **{lang.title()}**: {data['count']} patterns, {(data['avg_success'] * 100):.1f}% avg success\n"
    
    dashboard += "\n## Performance by Query Type\n"
    for qtype, data in sorted(stats.patterns_by_type.items(), key=lambda x: x[1]['count'], reverse=True):
        dashboard += f"- **{qtype.title()}**: {data['count']} patterns, {(data['avg_success'] * 100):.1f}% avg success\n"
    
    if stats.top_successful_patterns:
        dashboard += "\n## Top Successful Solution Patterns\n"
        for i, pattern in enumerate(stats.top_successful_patterns[:5], 1):
            dashboard += f"{i}. **{pattern['pattern']}** ({pattern['language']}, {pattern['type']})\n"
            dashboard += f"   - Success Rate: {(pattern['success_rate'] * 100):.1f}% ({pattern['uses']} uses)\n"
            dashboard += f"   - Source: {pattern['source']}\n"
            dashboard += f"   - Approach: {pattern['approach']}\n\n"
    
    return dashboard

@mcp.resource("adaptive-rag://knowledge-base/{language}")
def get_language_knowledge(language: str) -> str:
    """Get knowledge base summary for a specific programming language"""
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    c = conn.cursor()
    
    # Language-specific stats
    c.execute('''SELECT COUNT(*), AVG(avg_success_rate), SUM(success_count), SUM(failure_count)
                 FROM patterns WHERE language = ?''', (language,))
    row = c.fetchone()
    
    if not row or row[0] == 0:
        conn.close()
        return f"No knowledge found for {language} yet. Start using it and provide feedback to build the knowledge base!"
    
    total_patterns, avg_success, total_successes, total_failures = row
    
    # Patterns by source for this language
    c.execute('''SELECT source, COUNT(*) FROM patterns WHERE language = ? GROUP BY source''', (language,))
    sources = {row[0]: row[1] for row in c.fetchall()}
    
    # Top patterns for this language
    c.execute('''SELECT pattern_text, solution_approach, avg_success_rate, query_type, source,
                        success_count + failure_count as total_uses
                 FROM patterns
                 WHERE language = ? AND avg_success_rate > 0
                 ORDER BY avg_success_rate DESC, total_uses DESC
                 LIMIT 8''', (language,))
    
    top_patterns = c.fetchall()
    conn.close()
    
    knowledge_summary = f"""# {language.title()} Knowledge Base

## Overview
- **Total Patterns**: {total_patterns}
- **Average Success Rate**: {(avg_success or 0) * 100:.1f}%
- **Successful Solutions**: {total_successes or 0}
- **Learning Opportunities**: {total_failures or 0}

## Knowledge Sources
"""
    
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_patterns) * 100
        knowledge_summary += f"- **{source}**: {count} patterns ({percentage:.1f}%)\n"
    
    knowledge_summary += "\n## Most Effective Solution Patterns\n"
    
    for i, (pattern, solution, success_rate, query_type, source, uses) in enumerate(top_patterns, 1):
        knowledge_summary += f"""
### {i}. {pattern} ({query_type})
- **Success Rate**: {(success_rate * 100):.1f}% ({uses} uses)
- **Source**: {source}
- **Proven Approach**: {solution[:300] + '...' if solution and len(solution) > 300 else (solution or 'Approach being refined')}
"""
    
    return knowledge_summary

# --- Enhanced Prompts ---
@mcp.prompt()
def enhanced_coding_prompt(query: str, context_level: str = "detailed") -> str:
    """Create an enhanced coding prompt using adaptive RAG context."""
    enhanced_context = get_enhanced_context(query)
    
    if context_level == "brief":
        if enhanced_context.has_similar_patterns:
            best = enhanced_context.similar_patterns[0]
            return f"Context: Similar solutions exist with {best['success_rate']:.1%} success rate from {best['source']}.\n\nQuery: {query}"
        else:
            return f"Query: {query}"
    
    elif context_level == "detailed":
        return enhanced_context.suggested_context
    
    else:  # "comprehensive"
        prompt = enhanced_context.suggested_context
        if enhanced_context.has_similar_patterns:
            prompt += "\n\nAdditional successful patterns for reference:\n"
            for i, pattern in enumerate(enhanced_context.similar_patterns[:3], 1):
                prompt += f"{i}. {pattern['approach'][:150]}... (Success: {pattern['success_rate']:.1%}, Source: {pattern['source']})\n"
        
        return prompt

@mcp.prompt()
def knowledge_ingestion_prompt(content_type: str = "code") -> str:
    """Guide for ingesting different types of knowledge into the system."""
    
    if content_type == "code":
        return """I have some code that I'd like to add to the knowledge base. Here's how to do it:

1. **For individual functions/solutions**: Use `add_custom_knowledge` with question-answer pairs
2. **For code files**: Use `extract_knowledge_from_code` to automatically extract patterns
3. **For entire repositories**: Use `analyze_repository` to process multiple files

What type of code knowledge would you like to add?"""
    
    elif content_type == "documentation":
        return """I want to add documentation knowledge. You can:

1. Convert documentation sections into question-answer pairs
2. Use `add_custom_knowledge` with entries like:
   - Question: "How to set up authentication?"
   - Answer: [Your documentation content]
   - Source: "Company Documentation"

What documentation would you like to add?"""
    
    else:
        return """I can help you add various types of knowledge:

- **Code patterns**: Functions, classes, common solutions
- **Documentation**: Setup guides, best practices
- **Troubleshooting**: Error solutions, debugging steps
- **Company practices**: Internal standards, workflows

What type of knowledge would you like to add to the system?"""

if __name__ == "__main__":
    mcp.run()
