import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional, Dict
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
from datetime import datetime
import sqlite3
from contextlib import contextmanager
import asyncio
from threading import Lock
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict
from threading import Lock


class QueryRequest(BaseModel):
    query: str
    person_id: str
    topic_id: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.1

class QueryResponse(BaseModel):
    query: str
    answer: str
    person_id: str
    topic_id: str
    timestamp: datetime

class ConversationListResponse(BaseModel):
    person_id: str
    topic_id: str
    last_message: str
    last_updated: datetime
    message_count: int

# Base system prompt that will be combined with topic-specific prompts
BASE_SYSTEM_PROMPT = """You are Carrie Bot. Your job is to explore the user's stories to gather as much data as possible in a warm, curious way.
Start with 1 cocise question everytime, context rich and engaging question at a time. DO NOT ASK LONG QUESTIONS.

For example, if the topic is "My Story", the questions would be like;
"How would you like to be remembered by your loved ones?
Tell us about something or someone that makes you smile and why
Tell us about a family tradition that you'd like for your loved ones to continue
"

But now remember the topic is {topic} and you can ask one question at a time.
INSTRUCTIONS:
Be a good listner. 
Do not bore the user asking long questions.
Change the question if the current question becomes boring and too long (Your goal is to fetch as much memories as possible. Do not stick to one question for too long).
If the user ask question instead and it is not related to {topic}, you should say "I'm sorry, I can only inquire about  {topic} stories. Please tell me something related to that or you can change the topic in the previous page."
"""

# Topic-specific prompts
TOPIC_PROMPTS = {
    "funeral": """You are a compassionate and respectful virtual assistant (But you know your limits) designed to help users—primarily older adults—plan for their own funeral arrangements. 
    Your primary role is to ask thoughtful, gentle, and clear questions to understand the user's preferences.
    Always guide the conversation by asking one question at a time. Do not provide lengthy descriptions or explanations unless clarification is requested. 
    Your goal is to gather as much information as the user is comfortable sharing about their wishes, while making them feel safe, supported, and in control.
    Start by asking whether the user wants to be involved in their own funeral planning. 
    If they do, continue to ask relevant questions covering practical, emotional, and personal aspects of their funeral preferences. 
    Always remain calm, kind, and patient. Avoid assumptions, and adapt your next question based on the user's responses.
    Keep your responses brief, respectful, and always in the form of a question.""",

    "family": """You are a warm and understanding AI assistant helping to collect family memories and stories.
Focus on understanding family relationships, traditions, and meaningful moments.
Ask thoughtful questions about:
- Family traditions and rituals
- Important family events
- Relationships with family members
- Family history and heritage
- Special family memories""",

    "children": """You are a caring AI assistant helping to collect memories and messages for children.
Focus on understanding the person's relationship with their children and their hopes for them.
Ask thoughtful questions about:
- Special moments with children
- Parenting experiences
- Hopes and dreams for children
- Advice for children
- Family traditions with children""",

    "digital_assets": """You are a practical AI assistant helping to organize digital assets and online presence.
Focus on understanding the person's digital footprint and how they want it managed.
Ask thoughtful questions about:
- Social media accounts
- Digital photos and videos
- Online accounts and passwords
- Digital legacy preferences
- Important digital documents""",

    "best_memories": """You are a nostalgic AI assistant helping to collect cherished memories.
Focus on understanding the person's most meaningful life experiences.
Ask thoughtful questions about:
- Most cherished moments
- Life achievements
- Special relationships
- Travel experiences
- Personal milestones""",

    "friends": """You are a friendly AI assistant helping to collect memories about friendships.
Focus on understanding the person's relationships with friends and meaningful social connections.
Ask thoughtful questions about:
- Special friendships
- Shared experiences
- Friendship traditions
- Important social moments
- Messages for friends""",

    "childhood": """You are a warm AI assistant helping to collect childhood memories.
Focus on understanding the person's early life experiences and formative moments.
Ask thoughtful questions about:
- Early family life
- School experiences
- Childhood friends
- Special childhood memories
- Early dreams and aspirations"""
}

class DatabaseManager:
    def __init__(self, db_path: str = "chat_history.db"):
        self.db_path = db_path #
        self.lock = Lock() #
        self.init_database() #

    def init_database(self):
        """Initialize the database and create tables if they don't exist"""
        with self.get_connection() as conn: #
            cursor = conn.cursor() #
            
            # REMOVED: cursor.execute("DROP TABLE IF EXISTS conversations") 
            # This line was causing the table to be deleted on every restart.
            
            # Create conversations table with person_id and topic_id only if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    topic_id TEXT NOT NULL,
                    user_query TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """) #
            
            # Create indexes for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_person_topic 
                ON conversations(person_id, topic_id)
            """) #
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON conversations(timestamp)
            """) #
            
            conn.commit() #

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False) #
        conn.row_factory = sqlite3.Row #
        try:
            with self.lock: #
                yield conn #
        finally:
            conn.close() #

    def add_interaction(self, person_id: str, topic_id: str, user_query: str, assistant_response: str):
        """Add a new interaction to the database"""
        with self.get_connection() as conn: #
            cursor = conn.cursor() #
            cursor.execute("""
                INSERT INTO conversations (person_id, topic_id, user_query, assistant_response, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (person_id, topic_id, user_query, assistant_response, datetime.now())) #
            conn.commit() #

    def get_conversation_history(self, person_id: str, topic_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """Get conversation history for a specific person and topic"""
        with self.get_connection() as conn: #
            cursor = conn.cursor() #
            cursor.execute("""
                SELECT user_query, assistant_response, timestamp
                FROM conversations
                WHERE person_id = ? AND topic_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
            """, (person_id, topic_id, limit)) #
            
            rows = cursor.fetchall() #
            return [
                {
                    "query": row["user_query"], #
                    "response": row["assistant_response"], #
                    "timestamp": row["timestamp"] #
                }
                for row in rows
            ]

    def get_all_conversations(self) -> List[Dict]:
        """Get list of all conversations with summary info"""
        with self.get_connection() as conn: #
            cursor = conn.cursor() #
            cursor.execute("""
                SELECT 
                    person_id,
                    topic_id,
                    user_query as last_message,
                    timestamp as last_updated,
                    COUNT(*) as message_count
                FROM conversations
                WHERE id IN (
                    SELECT MAX(id)
                    FROM conversations
                    GROUP BY person_id, topic_id
                )
                GROUP BY person_id, topic_id
                ORDER BY timestamp DESC
            """) #
            
            rows = cursor.fetchall() #
            return [
                {
                    "person_id": row["person_id"], #
                    "topic_id": row["topic_id"], #
                    "last_message": row["last_message"][:100] + "..." if len(row["last_message"]) > 100 else row["last_message"], #
                    "last_updated": row["last_updated"], #
                    "message_count": row["message_count"] #
                }
                for row in rows
            ]

    def delete_conversation(self, person_id: str, topic_id: str):
        """Delete conversation history for a specific person and topic"""
        with self.get_connection() as conn: #
            cursor = conn.cursor() #
            cursor.execute("""
                DELETE FROM conversations
                WHERE person_id = ? AND topic_id = ?
            """, (person_id, topic_id)) #
            conn.commit() #
            return cursor.rowcount #

    def get_conversation_stats(self):
        """Get database statistics"""
        with self.get_connection() as conn: #
            cursor = conn.cursor() #
            
            # Total conversations
            cursor.execute("SELECT COUNT(DISTINCT person_id || topic_id) as total_conversations FROM conversations") #
            total_conversations = cursor.fetchone()["total_conversations"] #
            
            # Total messages
            cursor.execute("SELECT COUNT(*) as total_messages FROM conversations") #
            total_messages = cursor.fetchone()["total_messages"] #
            
            # Most recent activity
            cursor.execute("SELECT MAX(timestamp) as last_activity FROM conversations") #
            last_activity = cursor.fetchone()["last_activity"] #
            
            return {
                "total_conversations": total_conversations, #
                "total_messages": total_messages, #
                "last_activity": last_activity #
            }

app = FastAPI(
    title="PostScrypt Memory Collection API",
    description="A memory collection system using OpenAI API with persistent SQL-based conversation storage",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database manager
db_manager = DatabaseManager()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

# Ensure the API key is not hardcoded
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

# Ensure the API URL is not hardcoded
if OPENAI_API_URL is None:
    raise ValueError("OPENAI_API_URL environment variable is not set.")

async def call_openai_api(messages: List[Dict], max_tokens: int = 100, temperature: float = 0.1):
    """Call OpenAI API with the provided messages"""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(OPENAI_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            print(f"OpenAI API error: {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"OpenAI API error: {e.response.text}")
        except Exception as e:
            print(f"Error calling OpenAI API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calling OpenAI API: {str(e)}")

def get_system_prompt(topic_id: str) -> str:
    """Get the combined system prompt for a specific topic"""
    topic_prompt = TOPIC_PROMPTS.get(topic_id, TOPIC_PROMPTS["childhood"])
    base_prompt = BASE_SYSTEM_PROMPT.format(topic=topic_id)
    return f"{base_prompt}\n\n{topic_prompt}"

def build_messages(query: str, conversation_history: List[Dict[str, str]], topic_id: str) -> List[Dict]:
    """Build messages array for OpenAI API including system prompt and conversation history"""
    messages = [{"role": "system", "content": get_system_prompt(topic_id)}]
    
    # Add conversation history
    for interaction in conversation_history:
        messages.append({"role": "user", "content": interaction["query"]})
        messages.append({"role": "assistant", "content": interaction["response"]})
    
    # Add current query
    messages.append({"role": "user", "content": query})
    
    return messages

@app.post("/chat/query", response_model=QueryResponse)
async def chat_query(request: QueryRequest):
    """Main chat endpoint"""
    try:
        # Validate request
        if not request.person_id or not request.topic_id:
            raise HTTPException(status_code=400, detail="person_id and topic_id are required")
        
        # Get conversation history from database
        conversation_history = db_manager.get_conversation_history(request.person_id, request.topic_id)
        
        # Build messages for OpenAI API
        messages = build_messages(request.query, conversation_history, request.topic_id)
        
        # Call OpenAI API
        answer = await call_openai_api(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Store interaction in database
        db_manager.add_interaction(request.person_id, request.topic_id, request.query, answer)
        
        return QueryResponse(
            query=request.query,
            answer=answer,
            person_id=request.person_id,
            topic_id=request.topic_id,
            timestamp=datetime.now()
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in chat_query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/conversations", response_model=List[ConversationListResponse])
def get_all_conversations():
    """Get list of all conversations"""
    try:
        conversations = db_manager.get_all_conversations()
        return [
            ConversationListResponse(
                person_id=conv["person_id"],
                topic_id=conv["topic_id"],
                last_message=conv["last_message"],
                last_updated=datetime.fromisoformat(conv["last_updated"]) if isinstance(conv["last_updated"], str) else conv["last_updated"],
                message_count=conv["message_count"]
            )
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/conversation/{person_id}/{topic_id}")
def get_conversation_history(person_id: str, topic_id: str, limit: int = 50):
    """Get conversation history for a specific person and topic"""
    try:
        history = db_manager.get_conversation_history(person_id, topic_id, limit)
        return {
            "person_id": person_id,
            "topic_id": topic_id,
            "history": history,
            "total_messages": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/conversation/{person_id}/{topic_id}")
def delete_conversation(person_id: str, topic_id: str):
    """Delete conversation history for a specific person and topic"""
    try:
        deleted_count = db_manager.delete_conversation(person_id, topic_id)
        return {
            "message": f"Conversation for person {person_id} and topic {topic_id} deleted successfully",
            "deleted_messages": deleted_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/topics")
def get_topics():
    """Get list of available topics and their prompts"""
    return {
        "topics": list(TOPIC_PROMPTS.keys()),
        "prompts": TOPIC_PROMPTS
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        stats = db_manager.get_conversation_stats()
        return {
            "status": "healthy",
            "api_configured": OPENAI_API_KEY != "your-api-key-here",
            "database_connected": True,
            "stats": stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "api_configured": OPENAI_API_KEY != "your-api-key-here",
            "database_connected": False,
            "error": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1
    )