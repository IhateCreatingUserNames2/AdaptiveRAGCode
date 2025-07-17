# Adaptive RAG System for Code Intelligence

A self-learning retrieval system that helps AI coding assistants get smarter over time by remembering what solutions actually work.

## What This Does

Instead of just searching for similar-looking code snippets, this system learns from real coding successes and failures. When you ask "how do I fix this Python error?", it doesn't just find text that looks similar—it finds solutions that actually worked for similar problems in the past, along with their success rates.

Think of it as giving your AI coding assistant a memory that gets better with experience.

## The Two Main Components

### 🧠 ServerGemini.py - The Smart Memory
This is the "brain" that runs continuously in the background. It:
- **Receives your coding questions** and finds the most relevant solutions from past successes
- **Learns from feedback** when you tell it a solution worked (or didn't)
- **Gets smarter over time** by building a knowledge base of proven solutions
- **Filters by programming language** so Python questions get Python solutions

### 📚 generatorCheckPoint.py - The Knowledge Loader  
This is the "teacher" that gives the brain initial knowledge. It:
- **Loads the Microsoft NextCoder dataset** (380k+ coding examples)
- **Processes and stores them** as searchable memories
- **Runs once during setup** to populate the initial knowledge base
- **Can resume if interrupted** using checkpoints

## How It Could Help Coding Assistants

### Current Problem
When you ask Claude, Cursor, or other AI coding tools for help, they work from their training data alone. They can't learn from what actually worked in your specific context or remember successful solutions from your past projects.

### What This Adds
```
Your Question → Adaptive RAG → Enhanced Context → AI Assistant → Better Answer
```

Instead of:
```
"How do I fix this Python import error?"
→ Generic advice about import statements
```

You get:
```
"How do I fix this Python import error?"
→ "Based on 15 similar cases (94% success rate): The issue is usually relative imports. 
   Here's the exact solution that worked for others with this pattern..."
→ Much more targeted, proven solution
```

## Installation & Setup

### Prerequisites
```bash
pip install flask flask-cors sqlite3 chromadb sentence-transformers datasets pandas tqdm python-dotenv
```

### Step 1: Initial Knowledge Loading
```bash
# This runs once to populate the knowledge base (takes 2-4 hours)
python generatorCheckPoint.py
```

This will:
- Download the Microsoft NextCoder dataset
- Process it into searchable memories
- Create `adaptive_rag.db` (metadata) and `chroma_db_gte/` (vectors)
- Save progress in `ingestion_checkpoint.txt` (resumable if interrupted)

### Step 2: Start the Memory Service
```bash
# This runs continuously to serve queries
python ServerGemini.py
```

The service will start on `http://localhost:8002` with a simple dashboard.

### Step 3: Integration with Your AI Assistant

The system provides a REST API that any coding assistant can use:

#### Get Enhanced Context
```python
import requests

# Send your coding question
response = requests.post('http://localhost:8002/api/enhance', json={
    'query': 'How do I fix ModuleNotFoundError in Python?',
    'tool': 'vscode',
    'user_id': 'developer123'
})

enhanced_context = response.json()['suggested_context']
# Feed this to Claude/GPT/etc along with the original question
```

#### Provide Feedback
```python
# When a solution works
requests.post('http://localhost:8002/api/feedback', json={
    'pattern_id': enhanced_context['pattern_id'],
    'success': True,
    'solution_approach': 'Used absolute imports instead of relative'
})
```

## Integration Examples

### With Claude API
```python
import anthropic
import requests

def enhanced_claude_query(question):
    # Get enhanced context
    rag_response = requests.post('http://localhost:8002/api/enhance', 
                                json={'query': question})
    context = rag_response.json()['suggested_context']
    
    # Send to Claude with context
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        messages=[{"role": "user", "content": context}]
    )
    
    return response.content[0].text
```

### Browser Extension Integration
```javascript
// For web-based IDEs like Cursor, WindSurf
async function getEnhancedHelp(codeQuestion) {
    const response = await fetch('http://localhost:8002/api/enhance', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: codeQuestion,
            tool: 'cursor',
            user_id: 'current_user'
        })
    });
    
    const enhancement = await response.json();
    return enhancement.suggested_context;
}
```

## Expected Improvements

Based on our testing, integrating this system typically shows:

- **More specific solutions**: Instead of generic advice, get proven fixes for your exact error pattern
- **Better context awareness**: Solutions that worked in similar programming contexts
- **Learning from your patterns**: The system remembers what works in your specific development environment
- **Reduced trial-and-error**: Higher first-try success rates for debugging

## Dashboard & Monitoring

Visit `http://localhost:8002` to see:
- How many coding patterns the system has learned
- Success rates by programming language
- Most effective solution approaches
- System learning progress over time

## File Structure
```
adaptive-rag/
├── ServerGemini.py          # Main service (keep running)
├── generatorCheckPoint.py   # Initial setup (run once)
├── adaptive_rag.db          # Metadata storage
├── chroma_db_gte/           # Vector embeddings
├── ingestion_checkpoint.txt # Resume point for setup
└── requirements.txt         # Dependencies
```

## Limitations & Expectations

**This is experimental software.** We're sharing it because:
- It shows promising results in our testing
- The approach is novel and might inspire better implementations
- The coding community could benefit from adaptive learning systems

**Current limitations:**
- Requires local setup and maintenance
- Works best with common programming languages (Python, JavaScript, etc.)
- Needs feedback to improve (it's only as good as the solutions you validate)
- May not help with very cutting-edge or niche technologies

**Not a replacement for:** Good documentation, testing, code reviews, or learning fundamentals.

## Contributing

If you find this useful or have ideas for improvements:
- Test it with your coding workflow
- Share feedback about what works/doesn't work
- Suggest integration patterns for other tools
- Help optimize the learning algorithms

This is a research prototype that we hope sparks innovation in adaptive AI tools for developers.

## Cost & Performance

- **Setup cost**: ~$2 in OpenAI embedding API calls (one-time)
- **Runtime cost**: Minimal (mostly local processing)
- **Memory usage**: ~2-4GB for full dataset
- **Query speed**: <100ms for most lookups

---

*Built with the belief that AI coding assistants should learn from experience, not just training data.*
