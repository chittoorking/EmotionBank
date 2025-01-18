from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from models import Memory
from memory_handler import MemoryHandler
from emotion_tagging import EmotionAnalyzer
from ai_companion import AICompanion
from datetime import datetime
from database import SessionLocal, init_db
from typing import Optional
import uvicorn
import json

app = FastAPI()

# Initialize database session
db = SessionLocal()

# Initialize the database
init_db()

# Initialize handlers
memory_handler = MemoryHandler(db)
emotion_tagger = EmotionAnalyzer()
ai_companion = AICompanion()

@app.post("/upload_memory/")
async def upload_memory(
    caption: str = Form(...),
    content: str = Form(...),
    emotional_tags: str = Form(...),
    file: UploadFile = File(...)
):
    if not caption or not content or not emotional_tags:
        raise HTTPException(status_code=400, detail="Caption, content, and emotional tags are required.")

    try:
        # Parse emotional tags (assuming they're passed as a JSON string or comma-separated values)
        try:
            tags = json.loads(emotional_tags)
        except json.JSONDecodeError:
            tags = [tag.strip() for tag in emotional_tags.split(',')]

        # Create memory object
        memory = Memory(
            caption=caption,
            content=content,
            emotional_tags=tags,
            timestamp=datetime.now().isoformat()
        )
        
        # Upload memory and handle file
        result = await memory_handler.upload_memory(memory, file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading memory: {str(e)}")

@app.get("/retrieve_memory/")
async def retrieve_memory(query: str):
    return await memory_handler.retrieve_memory(query)

@app.get("/chat/")
async def chat_with_ai(user_input: str):
    return await ai_companion.chat(user_input)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)