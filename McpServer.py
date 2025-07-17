#!/usr/bin/env python3
"""
Adaptive RAG MCP Server

An MCP server that provides intelligent context retrieval and learning capabilities
for coding assistants. This server learns from successful solutions and provides
increasingly relevant context for coding queries.

Usage:
    # Development mode with MCP Inspector
    uv run mcp dev adaptive_rag_mcp_server.py

    # Install in Claude Desktop
    uv run mcp install adaptive_rag_mcp_server.py --name "Adaptive RAG"

    # Direct execution
    python adaptive_rag_mcp_server.py
"""

import os
import sqlite3
from datetime import datetime
import hashlib
import chromadb
from chromadb.utils import embedding_functions
import math
from typing import Dict, Any, List, Optional
import json

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base
from pydantic import BaseModel, Field


# --- Initialize ChromaDB and SQLite ---
def init_rag_system():
    """Initialize the RAG system components"""
    # ChromaDB setup
    chroma_client = chromadb.PersistentClient(path="chroma_db_gte")
    gte_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="thenlper/gte-large"
    )
    collection = chroma_client.get_or_create_collection(
        name="adaptive_rag_patterns_gte",
        embedding_function=gte_ef
    )

    # SQLite setup
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    c = conn.cursor()

    # Create tables if they don't exist
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


# Initialize the system
collection = init_rag_system()
print(f"ChromaDB collection loaded with {collection.count()} learned patterns.")

# Create the MCP server
mcp = FastMCP("AdaptiveRAG")


def extract_pattern_features(query: str) -> Dict[str, str]:
    """Extract key features from a coding query for pattern matching"""
    query_lower = query.lower()

    # Detect query type
    query_type = 'general'
    if any(word in query_lower for word in ['debug', 'error', 'fix', 'issue', 'problem', 'cannot', 'why']):
        query_type = 'debugging'
    elif any(word in query_lower for word in ['optimize', 'performance', 'faster', 'improve', 'slow']):
        query_type = 'optimization'
    elif any(word in query_lower for word in ['create', 'build', 'implement', 'make', 'how to', 'write']):
        query_type = 'creation'
    elif any(word in query_lower for word in ['refactor', 'clean', 'reorganize', 'best way']):
        query_type = 'refactoring'

    # Detect programming language
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

    # Create pattern text from important words
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


# --- MCP Tools ---

class EnhancedContext(BaseModel):
    """Enhanced context with similar solutions and success rates"""
    query: str = Field(description="The original query")
    has_similar_patterns: bool = Field(description="Whether similar patterns were found")
    suggested_context: str = Field(description="Enhanced context to provide to the LLM")
    pattern_id: int = Field(description="Internal pattern ID for feedback")
    similar_patterns: List[Dict[str, Any]] = Field(description="List of similar successful patterns")


@mcp.tool()
def get_enhanced_context(query: str, tool: str = "mcp_client", user_id: str = "anonymous") -> EnhancedContext:
    """
    Get enhanced context for a coding query by finding similar successful solutions.

    This tool searches the adaptive knowledge base for similar coding problems
    that have been successfully solved before, returning proven approaches
    along with their success rates.
    """
    # Extract features to detect language and query type
    features = extract_pattern_features(query)
    detected_lang = features.get('language', 'general')

    # Search ChromaDB for similar patterns
    query_params = {
        "query_texts": [query],
        "n_results": 5
    }
    if detected_lang != 'general':
        query_params["where"] = {"language": detected_lang}

    results = collection.query(**query_params)

    # Process results
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

    # Create or find pattern in SQLite for tracking
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

    # Log usage
    user_hash = hashlib.md5(user_id.encode()).hexdigest()[:16]
    c.execute('''INSERT INTO usage_logs (pattern_id, query, tool, user_hash, created_at)
                 VALUES (?, ?, ?, ?, ?)''',
              (pattern_id, query, tool, user_hash, datetime.now()))

    conn.commit()
    conn.close()

    # Build enhanced context
    if similar_patterns:
        best_pattern = similar_patterns[0]
        suggested_context = f"""Based on a similar problem (similarity: {best_pattern['similarity']:.2f}, success rate: {best_pattern['success_rate']:.1%}):
- Past successful approach: {best_pattern['approach']}

Additional context from {len(similar_patterns)} similar cases:
""" + "\n".join([f"- {p['approach'][:100]}..." if len(p['approach']) > 100 else f"- {p['approach']}" for p in
                 similar_patterns[1:3]])

        suggested_context += f"\n\nHere is the user's question, please use this context to provide a better answer:\n---\n{query}"
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
    """Result of recording feedback"""
    status: str = Field(description="Status of the feedback recording")
    learned: bool = Field(description="Whether the system learned from this feedback")
    pattern_id: int = Field(description="Pattern ID that was updated")


@mcp.tool()
def record_solution_feedback(
        pattern_id: int,
        query: str,
        solution_approach: str,
        success: bool
) -> FeedbackResult:
    """
    Record feedback about whether a solution worked or not.

    This is crucial for the adaptive learning - when you tell the system
    that a solution worked (or didn't), it learns and improves future
    recommendations for similar problems.
    """
    conn = sqlite3.connect('adaptive_rag.db', timeout=10)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    try:
        # Update success/failure counts
        if success:
            c.execute('''UPDATE patterns SET success_count = success_count + 1, 
                        solution_approach = ?, updated_at = ? WHERE id = ?''',
                      (solution_approach[:500], datetime.now(), pattern_id))
        else:
            c.execute('''UPDATE patterns SET failure_count = failure_count + 1, 
                        updated_at = ? WHERE id = ?''',
                      (datetime.now(), pattern_id))

        # Recalculate success rate
        c.execute('''UPDATE patterns SET avg_success_rate = 
                    CAST(success_count AS FLOAT) / (success_count + failure_count)
                    WHERE id = ? AND (success_count + failure_count) > 0''', (pattern_id,))

        # Update latest usage log
        c.execute('''UPDATE usage_logs SET success = ? WHERE id = (
                       SELECT id FROM usage_logs WHERE pattern_id = ? 
                       ORDER BY created_at DESC LIMIT 1)''',
                  (success, pattern_id))

        learned = False
        # If successful, add to ChromaDB knowledge base
        if success and query and solution_approach:
            c.execute('SELECT * FROM patterns WHERE id = ?', (pattern_id,))
            pattern_stats = c.fetchone()

            if pattern_stats:
                doc_id = f"pattern_{pattern_id}"
                metadata_dict = dict(pattern_stats)

                # Convert datetime objects to strings for ChromaDB
                for key, value in metadata_dict.items():
                    if isinstance(value, datetime):
                        metadata_dict[key] = str(value)

                # Store the successful solution in ChromaDB
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

    return FeedbackResult(
        status=status,
        learned=learned,
        pattern_id=pattern_id
    )


class SystemStats(BaseModel):
    """Statistics about the adaptive RAG system"""
    total_patterns: int = Field(description="Total patterns in SQLite database")
    learned_vectors: int = Field(description="Total learned vectors in ChromaDB")
    patterns_by_language: Dict[str, Dict[str, Any]] = Field(description="Patterns grouped by programming language")
    patterns_by_type: Dict[str, Dict[str, Any]] = Field(description="Patterns grouped by query type")
    top_successful_patterns: List[Dict[str, Any]] = Field(description="Most successful patterns")


@mcp.tool()
def get_system_statistics() -> SystemStats:
    """
    Get comprehensive statistics about the adaptive RAG system's learning progress.

    Shows how many patterns have been learned, success rates by programming language,
    and the most successful solution approaches.
    """
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

    # Top successful patterns
    c.execute('''SELECT pattern_text, solution_approach, avg_success_rate, 
                        success_count + failure_count as total_uses, language, query_type
                 FROM patterns
                 WHERE avg_success_rate > 0.8 AND total_uses > 2
                 ORDER BY avg_success_rate DESC, total_uses DESC
                 LIMIT 10''')
    top_patterns = []
    for row in c.fetchall():
        top_patterns.append({
            'pattern': row[0],
            'approach': row[1] or 'Not specified',
            'success_rate': row[2],
            'uses': row[3],
            'language': row[4],
            'type': row[5]
        })

    conn.close()

    return SystemStats(
        total_patterns=total_patterns,
        learned_vectors=collection.count(),
        patterns_by_language=patterns_by_language,
        patterns_by_type=patterns_by_type,
        top_successful_patterns=top_patterns
    )


# --- MCP Resources ---

@mcp.resource("adaptive-rag://dashboard")
def get_dashboard() -> str:
    """Real-time dashboard showing system learning progress"""
    stats = get_system_statistics()

    dashboard = f"""# Adaptive RAG System Dashboard

## Learning Progress
- **Total Patterns Tracked**: {stats.total_patterns}
- **Learned Solutions**: {stats.learned_vectors}
- **Knowledge Base Growth**: {((stats.learned_vectors / max(stats.total_patterns, 1)) * 100):.1f}% patterns have successful solutions

## Performance by Programming Language
"""

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
            dashboard += f"   - Approach: {pattern['approach'][:100]}...\n\n"

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

    # Top patterns for this language
    c.execute('''SELECT pattern_text, solution_approach, avg_success_rate, query_type,
                        success_count + failure_count as total_uses
                 FROM patterns
                 WHERE language = ? AND avg_success_rate > 0
                 ORDER BY avg_success_rate DESC, total_uses DESC
                 LIMIT 5''', (language,))

    top_patterns = c.fetchall()
    conn.close()

    knowledge_summary = f"""# {language.title()} Knowledge Base

## Overview
- **Total Patterns**: {total_patterns}
- **Average Success Rate**: {(avg_success or 0) * 100:.1f}%
- **Successful Solutions**: {total_successes or 0}
- **Learning Opportunities**: {total_failures or 0}

## Most Effective Solution Patterns
"""

    for i, (pattern, solution, success_rate, query_type, uses) in enumerate(top_patterns, 1):
        knowledge_summary += f"""
### {i}. {pattern} ({query_type})
- **Success Rate**: {(success_rate * 100):.1f}% ({uses} uses)
- **Proven Approach**: {solution or 'Approach being refined'}
"""

    return knowledge_summary


# --- MCP Prompts ---

@mcp.prompt()
def enhanced_coding_prompt(query: str, context_level: str = "detailed") -> str:
    """
    Create an enhanced coding prompt using adaptive RAG context.

    This prompt template automatically retrieves relevant context from past
    successful solutions and formats it for optimal LLM performance.
    """
    enhanced_context = get_enhanced_context(query)

    if context_level == "brief":
        if enhanced_context.has_similar_patterns:
            return f"Context: Similar solutions exist with {enhanced_context.similar_patterns[0]['success_rate']:.1%} success rate.\n\nQuery: {query}"
        else:
            return f"Query: {query}"

    elif context_level == "detailed":
        return enhanced_context.suggested_context

    else:  # "comprehensive"
        prompt = enhanced_context.suggested_context
        if enhanced_context.has_similar_patterns:
            prompt += "\n\nAdditional successful patterns for reference:\n"
            for i, pattern in enumerate(enhanced_context.similar_patterns[:3], 1):
                prompt += f"{i}. {pattern['approach'][:150]}... (Success: {pattern['success_rate']:.1%})\n"

        return prompt


@mcp.prompt()
def debug_assistant_prompt(error_message: str, code_context: str = "") -> List[base.Message]:
    """
    Create a debugging prompt with enhanced context from similar error resolutions.
    """
    debug_query = f"debug error: {error_message}"
    if code_context:
        debug_query += f" in context: {code_context[:200]}"

    enhanced_context = get_enhanced_context(debug_query)

    messages = [
        base.UserMessage("I'm encountering this error:"),
        base.UserMessage(error_message),
    ]

    if code_context:
        messages.append(base.UserMessage(f"Code context:\n```\n{code_context}\n```"))

    if enhanced_context.has_similar_patterns:
        best_solution = enhanced_context.similar_patterns[0]
        messages.append(base.AssistantMessage(
            f"I see this type of error. Based on previous successful resolutions "
            f"(success rate: {best_solution['success_rate']:.1%}), here's what typically works:\n\n"
            f"{best_solution['approach']}"
        ))
        messages.append(base.UserMessage("Can you help me apply this solution to my specific case?"))
    else:
        messages.append(base.AssistantMessage(
            "This appears to be a new type of error in our knowledge base. "
            "Let me help you debug it step by step."
        ))

    return messages


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()